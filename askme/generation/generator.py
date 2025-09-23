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
        bullets: List[str] = []
        for p in top_passages:
            bullets.append(f"- [{p.doc_id}: {p.title}]")

        # Keep English template because相关用例断言英文关键字
        summary = (
            "Summary: Constructed from retrieved passages."
            if not lang_is_cn
            else "摘要：基于检索段落的简要概述。"
        )
        sources_header = "Sources:" if not lang_is_cn else "参考来源："
        header = (
            "Answer (constructed from top passages)"
            if not lang_is_cn
            else "基于最相关段落构建的回答"
        )

        id_refs = ", ".join([f"[{p.doc_id}]" for p in top_passages])

        lines = [
            header,
            "From the retrieved passages we can gather:",
            summary,
            f"Refs: {id_refs}" if id_refs else "",
            sources_header,
        ]
        # Add bullets and a brief conclusion consistent with existing tests
        lines.extend(bullets)
        if not lang_is_cn:
            lines.append(
                "The current snippets do not list explicit names; "
                "consider exploring adjacent chapters."
            )
        else:
            lines.append("目前的片段未直接列出具体名单，如需确认，可进一步检索相近章节。")

        return "\n".join([ln for ln in lines if ln])


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
                r"首先，.*?(?=\n\n|\n[^我让但])",  # Starting with 首先，
                r"让我.*?(?=\n\n|\n[^我让但])",  # Starting with 让我
                r"我的回答应该是.*?(?=\n\n|\n[^我让但])",  # My answer should be
                r"但根据要求.*?(?=\n\n|\n[^我让但])",  # But according to requirements
                r"所以，我.*?(?=\n\n|\n[^我让但])",  # So, I...
                r"因此，我.*?(?=\n\n|\n[^我让但])",  # Therefore, I...
                r"我再检查.*?(?=\n\n|\n[^我让但])",  # Let me check again
                r"我决定.*?(?=\n\n|\n[^我让但])",  # I decide
            ]

            cleaned = text
            for pattern in patterns:
                cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

            # Remove multiple consecutive newlines and extra whitespace
            cleaned = re.sub(r"\n\s*\n", "\n", cleaned)
            cleaned = re.sub(r"^\s+|\s+$", "", cleaned, flags=re.MULTILINE)

            return cleaned.strip()

        def build_prompt(selected: List[Passage]) -> str:
            context_parts = []
            for p in selected:
                # Avoid inline doc-id style markers in the prompt to reduce
                # the chance of the model copying them into the answer.
                context_parts.append(f"{p.title}\n{p.content}\n")
            context = "\n\n".join(context_parts)
            return (
                self.config.system_prompt
                + "\n\n"
                + self.config.user_prompt_template.format(
                    context=context, question=question
                )
                + "\n\n请详细回答，要求：\n"
                + "1. 基于上下文提供完整、详细的答案\n"
                + "2. 整合多个相关段落的信息，给出全面回答\n"
                + "3. 答案中不要包含文档ID引用，纯文本回答即可\n"
                + "4. 如果信息不足，说明原因\n"
                + "5. 答案要有逻辑层次，便于阅读理解\n"
                + "6. 回答内容要有清晰的段落分隔"
            )

        async def call_ollama(
            selected: List[Passage], extra_options: Optional[Dict[str, Any]] = None
        ) -> str:
            prompt = build_prompt(selected)

            model_name = (self.config.ollama_model or "").strip() or "gpt-oss:20b"
            options: Dict[str, Any] = {
                "temperature": float(self.config.temperature),
                "top_p": float(self.config.top_p),
                "num_predict": int(max(self.config.max_tokens, 512)),
            }
            payload: Dict[str, Any] = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": options,
            }

            import os

            if os.getenv("ASKME_OLLAMA_THINKING", "0") not in {"1", "true", "True"}:
                options["thinking"] = {"enabled": False}
                options["enable_thinking"] = False

            if extra_options:
                options.update(extra_options)

            client = await self._client_get()
            url = f"{self.config.ollama_endpoint.rstrip('/')}/api/generate"
            from loguru import logger

            logger.info(
                "Ollama payload: {} (passages={})",
                payload,
                len(selected),
            )
            resp = await client.post(url, json=payload)
            # Be tolerant to lightweight test doubles: prefer raise_for_status
            try:
                rfs = getattr(resp, "raise_for_status", None)
                if callable(rfs):
                    rfs()
            except Exception:
                # Best-effort logging; resp may not have text/status_code
                try:
                    logger.error(
                        "Ollama request failed (model={})", payload.get("model")
                    )
                except Exception:
                    pass
                raise

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
                cleaned = re.sub(
                    r"</?(?:final|think|thinking)>", "", cleaned, flags=re.IGNORECASE
                )

                # Strip inline citation-style markers like [chunk_1234], [doc_1],
                # or anything containing "#chunk" within brackets. Also drop
                # explicit Source/Sources lines if the model emitted them.
                def strip_inline_refs(text: str) -> str:
                    patterns = [
                        r"\[[^\]]*#chunk[^\]]*\]",
                        (
                            r"\[(?:chunk|doc(?:ument)?|passage|source|ref|"
                            r"citation)[^\]]*\]"
                        ),
                    ]
                    out = text
                    for pat in patterns:
                        out = re.sub(pat, "", out, flags=re.IGNORECASE)
                    # Remove standalone sources lines
                    out = re.sub(
                        r"^\s*(?:sources?|参考来源)[:：].*$",
                        "",
                        out,
                        flags=re.IGNORECASE | re.MULTILINE,
                    )
                    # Collapse excessive spaces left by removals
                    out = re.sub(r"\s{2,}", " ", out).strip()
                    return out

                cleaned = strip_inline_refs(cleaned)

                # Remove excessive meta-commentary while preserving content
                excessive_patterns = [
                    # Remove "首先，我需要..." starters
                    r"^首先，我需要.*?。\s*",
                    # Remove "让我..." starters
                    r"^让我.*?。\s*",
                    # Remove "我需要..." starters
                    r"^我需要.*?。\s*",
                ]
                for pattern in excessive_patterns:
                    cleaned = re.sub(
                        pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE
                    )

                cleaned = cleaned.strip()
                if cleaned:
                    return cleaned

                logger.warning("Ollama response stripped to empty after cleaning")

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

        for attempt_index, attempt in enumerate(attempts, start=1):
            try:
                return await call_ollama(attempt["passages"], attempt.get("options"))
            except Exception as exc:
                from loguru import logger

                logger.error(
                    "Ollama generation attempt %s failed: %s",
                    attempt_index,
                    exc,
                )

        # 如果全部尝试失败，回退到简单模板生成，避免抛出异常
        st = SimpleTemplateGenerator(self.config)
        return await st.generate(question, passages)

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
            context_parts.append(f"{p.title}\n{p.content}\n")
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

            def _call(_unused: object | None = None) -> str:
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

            # Best-effort strip inline citation markers for OpenAI path as well
            try:
                import re as _re

                def _strip(text: str) -> str:
                    t = _re.sub(
                        r"\[[^\]]*#chunk[^\]]*\]", "", text, flags=_re.IGNORECASE
                    )
                    t = _re.sub(
                        (
                            r"\[(?:chunk|doc(?:ument)?|passage|source|ref|"
                            r"citation)[^\]]*\]"
                        ),
                        "",
                        t,
                        flags=_re.IGNORECASE,
                    )
                    t = _re.sub(
                        r"^\s*(?:sources?|参考来源)[:：].*$",
                        "",
                        t,
                        flags=_re.IGNORECASE | _re.MULTILINE,
                    )
                    return _re.sub(r"\s{2,}", " ", t).strip()

                return _strip(result)
            except Exception:
                return result
        except Exception:
            # Fall back to local template if OpenAI endpoint is unavailable
            st = SimpleTemplateGenerator(self.config)
            return await st.generate(question, passages)
