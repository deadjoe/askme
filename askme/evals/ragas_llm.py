"""Custom Ragas LLM wrapper backed by local Ollama via OpenAI-compatible API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import openai
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.prompt_values import PromptValue

try:
    from ragas.llms.base import BaseRagasLLM as BaseRagasLLMBase
    from ragas.run_config import RunConfig as RunConfigType
except Exception:  # pragma: no cover - used in tests with stubbed ragas

    @dataclass
    class _FallbackRunConfig:
        timeout: int = 60

    class _FallbackBaseRagasLLM:  # minimal stub
        def __init__(
            self, run_config: _FallbackRunConfig | None = None, **kwargs: Any
        ) -> None:
            self.run_config = run_config or _FallbackRunConfig()

        def set_run_config(self, run_config: _FallbackRunConfig) -> None:
            self.run_config = run_config

        def __post_init__(self) -> None:
            pass

    BaseRagasLLMBase = _FallbackBaseRagasLLM
    RunConfigType = _FallbackRunConfig


def _to_openai_messages(prompt: PromptValue) -> List[Dict[str, str]]:
    base_messages: List[BaseMessage] = prompt.to_messages()
    formatted: List[Dict[str, str]] = []
    for message in base_messages:
        role = message.type
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"
        formatted.append({"role": role, "content": message.content})
    return formatted


@dataclass
class OllamaRagasLLM(BaseRagasLLMBase):
    model: str = "gpt-oss:20b"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama-local"

    def __post_init__(self) -> None:
        super().__post_init__()
        self._client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)

    # BaseRagasLLM expects synchronous + async APIs.
    # Reuse the sync call for async execution via a thread pool.
    def _generate(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: Optional[List[str]] = None,
    ) -> LLMResult:
        messages = _to_openai_messages(prompt)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            n=n,
            stop=stop,
        )

        generations: List[ChatGeneration] = []
        for choice in response.choices:
            text = choice.message.content or ""
            ai_msg = AIMessage(content=text)
            generations.append(ChatGeneration(message=ai_msg, text=text))

        llm_output: Dict[str, Any] = {}
        usage = getattr(response, "usage", None)
        if usage is not None:
            token_usage: Dict[str, Any]
            if hasattr(usage, "model_dump"):
                token_usage = usage.model_dump()
            elif isinstance(usage, dict):
                token_usage = usage
            else:
                token_usage = getattr(usage, "__dict__", {})
            llm_output["token_usage"] = token_usage

        return LLMResult(generations=[generations], llm_output=llm_output)

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: Optional[List[str]] = None,
        callbacks: Any = None,
    ) -> LLMResult:
        return self._generate(prompt, n=n, temperature=temperature, stop=stop)

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: Optional[float] = 0.01,
        stop: Optional[List[str]] = None,
        callbacks: Any = None,
    ) -> LLMResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._generate,
            prompt,
            n,
            temperature or self.get_temperature(n),
            stop,
        )

    def is_finished(self, response: LLMResult) -> bool:
        return all(gen.text.strip() for gens in response.generations for gen in gens)


def build_ollama_llm(settings: Dict[str, str]) -> OllamaRagasLLM:
    return OllamaRagasLLM(
        model=settings.get("model", "gpt-oss:20b"),
        base_url=settings.get("base_url", "http://localhost:11434/v1"),
        api_key=settings.get("api_key", "ollama-local"),
        run_config=RunConfigType(timeout=60),
    )
