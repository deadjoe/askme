"""Custom Ragas LLM wrapper backed by local Ollama via OpenAI-compatible API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, cast

import openai
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult
from langchain_core.prompt_values import PromptValue

if TYPE_CHECKING:
    from ragas.llms.base import BaseRagasLLM as BaseRagasLLMBase
    from ragas.run_config import RunConfig as RunConfigType
else:  # pragma: no cover - runtime import with fallback
    try:
        from ragas.llms.base import BaseRagasLLM as BaseRagasLLMBase
        from ragas.run_config import RunConfig as RunConfigType
    except Exception:

        @dataclass
        class RunConfigType:
            timeout: int = 60

        class BaseRagasLLMBase:  # minimal stub
            def __init__(
                self, run_config: RunConfigType | None = None, **kwargs: Any
            ) -> None:
                self.run_config = run_config or RunConfigType()

            def set_run_config(self, run_config: RunConfigType) -> None:
                self.run_config = run_config

            def __post_init__(self) -> None:
                pass


def _to_openai_messages(prompt: PromptValue) -> List[Dict[str, Any]]:
    base_messages: List[BaseMessage] = prompt.to_messages()
    formatted: List[Dict[str, Any]] = []
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
        self._client: openai.OpenAI = openai.OpenAI(
            base_url=self.base_url, api_key=self.api_key
        )

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
            messages=cast(Iterable[Any], messages),
            temperature=temperature,
            n=n,
            stop=stop,
        )

        generations: List[Generation] = []
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

        generation_rows: List[List[Generation]] = [generations]
        return LLMResult(generations=generation_rows, llm_output=llm_output)

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
        resolved_temp = temperature if temperature is not None else 0.01
        return await loop.run_in_executor(
            None,
            self._generate,
            prompt,
            n,
            resolved_temp,
            stop,
        )

    def is_finished(self, response: LLMResult) -> bool:
        return all(gen.text.strip() for gens in response.generations for gen in gens)


def build_ollama_llm(settings: Dict[str, str]) -> OllamaRagasLLM:
    llm = OllamaRagasLLM(
        model=settings.get("model", "gpt-oss:20b"),
        base_url=settings.get("base_url", "http://localhost:11434/v1"),
        api_key=settings.get("api_key", "ollama-local"),
    )
    try:
        llm.set_run_config(RunConfigType(timeout=60))
    except Exception:
        pass
    return llm
