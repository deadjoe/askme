"""
LLM generation interface with a simple local template generator and
an optional Local Ollama generator.

The Ollama generator uses the local HTTP API (default http://localhost:11434)
and is only used when explicitly enabled via config or env.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

import httpx

from askme.core.config import GenerationConfig

# OpenAI client是可选依赖，这里延迟导入并在缺失时提供显式错误
try:  # pragma: no cover - import guard
    _openai_module = importlib.import_module("openai")
    OpenAI: Any = getattr(_openai_module, "OpenAI")
except Exception:  # pragma: no cover
    OpenAI = None


@dataclass
class Passage:
    doc_id: str
    title: str
    content: str
    score: float


class BaseGenerator(ABC):
    def __init__(self, config: GenerationConfig):
        self.config = config

    @abstractmethod
    async def generate(self, question: str, passages: List[Passage]) -> str:
        """Generate an answer string given a question and supporting passages."""
        raise NotImplementedError

    async def cleanup(self) -> None:  # optional
        return None


class SimpleTemplateGenerator(BaseGenerator):
    """A minimal, offline generator that builds an answer from passages."""

    async def generate(self, question: str, passages: List[Passage]) -> str:
        if not passages:
            return (
                f"I could not find relevant information to answer: '{question}'. "
                "Please provide more context."
            )

        def is_chinese(text: str) -> bool:
            return any("\u4e00" <= ch <= "\u9fff" for ch in text)

        lang_is_cn = is_chinese(question)

        def compact(text: str, limit: int = 160) -> str:
            cleaned = " ".join(text.split())
            return cleaned[:limit] + ("…" if len(cleaned) > limit else "")

        top_passages = passages[: min(3, len(passages))]
        insight_lines: List[str] = []
        for p in top_passages:
            snippet = compact(p.content)
            insight_lines.append(f"[{p.doc_id}] {snippet}")

        if lang_is_cn:
            intro = "根据检索到的段落，目前可确认的信息如下："
            outro = (
                "若需更精确的答案，可以尝试换用其他关键词或补充更多上下文。"
            )
        else:
            intro = "From the retrieved passages we can gather:";
            outro = "Try refining the query or adding detail if you need more specifics."

        body = "\n".join(insight_lines)

        if not body:
            body = "-"  # 保底，理论上不会出现因为 passages 非空

        if lang_is_cn:
            conclusion = (
                "目前的片段未直接列出具体名单，如需确认，可进一步检索相近章节。"
            )
        else:
            conclusion = (
                "The current snippets do not list explicit names; consider exploring adjacent chapters."
            )

        return "\n".join([intro, body, conclusion, outro])


class LocalOllamaGenerator(BaseGenerator):
    """Call a local Ollama server to generate an answer."""

    def __init__(self, config: GenerationConfig):
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None

    async def _client_get(self) -> httpx.AsyncClient:
        if self._client is None:
            # Respect a longer read timeout for local large models
            import os

            read_t = float(os.getenv("ASKME_OLLAMA_READ_TIMEOUT", "120"))
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=read_t))
        return self._client

    async def generate(self, question: str, passages: List[Passage]) -> str:
        # Build prompt from system + concatenated passages
        def strip_think(text: str) -> str:
            # Remove various thinking patterns used by Qwen and similar models
            patterns = [
                r"<think>.*?</think>",  # Standard think tags
                r"<thinking>.*?</thinking>",  # Alternative thinking tags
                r"让我.*?思考.*?(?=\n|\s|$)",  # Chinese thinking patterns
                r"我需要.*?分析.*?(?=\n|\s|$)",  # Analysis patterns
                r"首先.*?考虑.*?(?=\n|\s|$)",  # First consider patterns
                r"标签中.*?说明.*?(?=\n|\s|$)",  # Tag explanation patterns
            ]

            cleaned = text
            for pattern in patterns:
                cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

            # Remove multiple consecutive newlines and extra whitespace
            cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
            cleaned = re.sub(r'^\s+|\s+$', '', cleaned, flags=re.MULTILINE)

            return cleaned.strip()

        def build_prompt(selected: List[Passage]) -> str:
            context_parts = []
            for p in selected:
                context_parts.append(f"[{p.doc_id}: {p.title}]\n{p.content}\n")
            context = "\n\n".join(context_parts)
            return (
                self.config.system_prompt
                + "\n\n"
                + self.config.user_prompt_template.format(
                    context=context, question=question
                )
                + "\n\n请简洁回答，要求：\n"
                + "1. 基于上下文直接给出答案，避免分析过程\n"
                + "2. 答案要具体明确，引用相关文档ID\n"
                + "3. 如果信息不足，说明原因\n"
                + "4. 回答控制在2-3句话内"
            )

        async def call_ollama(selected: List[Passage], extra_options: Optional[Dict[str, Any]] = None) -> str:
            prompt = build_prompt(selected)

            model_name = (self.config.ollama_model or "").strip() or "gpt-oss:20b"
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": float(self.config.temperature),
                    "top_p": float(self.config.top_p),
                    "num_predict": int(max(self.config.max_tokens, 512)),
                },
            }

            import os

            if os.getenv("ASKME_OLLAMA_THINKING", "0") not in {"1", "true", "True"}:
                payload["options"]["thinking"] = {"enabled": False}
                payload["options"]["enable_thinking"] = False

            if extra_options:
                payload["options"].update(extra_options)

            client = await self._client_get()
            url = f"{self.config.ollama_endpoint.rstrip('/')}/api/generate"
            from loguru import logger

            logger.info(
                "Ollama payload: {} (passages={})",
                payload,
                len(selected),
            )
            resp = await client.post(url, json=payload)
            if resp.status_code >= 400:
                logger.error(
                    "Ollama request failed: {} {} (model={})",
                    resp.status_code,
                    resp.text,
                    payload.get("model"),
                )
                resp.raise_for_status()

            try:
                data = resp.json()
            except Exception as json_err:
                logger.error("Failed to parse Ollama response: {}", json_err)
                logger.debug("Ollama raw response: {}", resp.text)
                raise

            answer = data.get("response") or data.get("message")
            if answer:
                answer_str = str(answer)

                # Clean up thinking patterns and unwanted content
                cleaned = strip_think(answer_str)

                # Remove any remaining XML-like tags
                cleaned = re.sub(r"</?(?:final|think|thinking)>", "", cleaned, flags=re.IGNORECASE)

                # Light cleanup: remove excessive meta-commentary while preserving content
                excessive_patterns = [
                    r"^首先，我需要.*?。\s*",  # Remove "首先，我需要..." starters
                    r"^让我.*?。\s*",  # Remove "让我..." starters
                    r"^我需要.*?。\s*",  # Remove "我需要..." starters
                ]
                for pattern in excessive_patterns:
                    cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)

                cleaned = cleaned.strip()
                if cleaned:
                    return cleaned

                logger.warning(
                    "Ollama response stripped to empty after cleaning"
                )

            logger.warning("Ollama returned empty response: {}", data)
            raise RuntimeError("ollama-empty-response")


        # primary attempt uses最多 8 段上下文，若失败再尝试仅保留最相关的 3 段
        attempts: List[Dict[str, Any]] = [
            {"passages": passages[:8], "options": {}},
        ]
        if len(passages) > 3:
            attempts.append({"passages": passages[:3], "options": {}})
        attempts.append(
            {
                "passages": passages[:3] if len(passages) > 3 else passages[:1],
                "options": {"num_predict": int(max(self.config.max_tokens, 1024))},
            }
        )

        last_error: Optional[Exception] = None
        for attempt_index, attempt in enumerate(attempts, start=1):
            try:
                return await call_ollama(
                    attempt["passages"], attempt.get("options")
                )
            except Exception as exc:
                from loguru import logger

                last_error = exc
                logger.error(
                    "Ollama generation attempt %s failed: %s",
                    attempt_index,
                    exc,
                )

        # 如果全部尝试失败，抛出最后一次的异常供上层兜底处理
        assert last_error is not None
        raise last_error

    async def cleanup(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class OpenAIChatGenerator(BaseGenerator):
    """Generator using OpenAI-compatible Chat Completions API.

    - Respects base_url for third-party OpenAI-compatible endpoints.
    - Reads API key from env using config.openai_api_key_env.
    """

    def __init__(self, config: GenerationConfig):
        super().__init__(config)
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        global OpenAI

        if OpenAI is None:
            raise RuntimeError(
                "OpenAI client library is not available; "
                "install openai or disable OpenAI generator"
            )
        if self._client is None:
            import os

            api_key = os.getenv(self.config.openai_api_key_env, "")
            self._client = OpenAI(api_key=api_key, base_url=self.config.openai_base_url)
        return self._client

    async def generate(self, question: str, passages: List[Passage]) -> str:
        # Build messages with system + user including references
        context_parts = []
        for p in passages[:8]:
            context_parts.append(f"[{p.doc_id}: {p.title}]\n{p.content}\n")
        context = "\n\n".join(context_parts) or "No context provided."

        messages = [
            {"role": "system", "content": self.config.system_prompt.strip()},
            {
                "role": "user",
                "content": self.config.user_prompt_template.format(
                    context=context, question=question
                ),
            },
        ]

        try:
            # Note: openai python SDK is sync; run in thread to avoid blocking loop
            import anyio

            def _call() -> str:
                client = self._get_client()
                resp = client.chat.completions.create(
                    model=self.config.openai_model,
                    messages=cast(List[Dict[str, Any]], messages),
                    temperature=float(self.config.temperature),
                    top_p=float(self.config.top_p),
                    max_tokens=int(self.config.max_tokens),
                )
                choice = resp.choices[0].message.content if resp.choices else ""
                return choice or ""

            result: str = await anyio.to_thread.run_sync(_call)
            return result
        except Exception:
            # Fall back to local template if OpenAI endpoint is unavailable
            st = SimpleTemplateGenerator(self.config)
            return await st.generate(question, passages)
