#!/usr/bin/env python3
"""Debug script for testing the Ollama generator locally."""

import asyncio
import traceback

from askme.core.config import get_settings
from askme.generation.generator import LocalOllamaGenerator, Passage


async def test_ollama() -> None:
    print("🔍 Testing Ollama Generator...")

    settings = get_settings()
    generator = LocalOllamaGenerator(settings.generation)

    passages = [
        Passage(
            doc_id="test-1",
            title="凡人修仙传",
            content="韩立是一个修仙者，从青牛镇出发开始修仙之路。",
            score=1.0,
        ),
        Passage(
            doc_id="test-2",
            title="修炼历程",
            content="他通过努力修炼，从炼气期开始，逐步提升修为。",
            score=0.9,
        ),
    ]

    question = "韩立是谁？"

    try:
        print(f"📝 Question: {question}")
        print(f"📚 Passages: {len(passages)} documents")
        print(
            f"🔧 Config: {settings.generation.ollama_model} @ "
            f"{settings.generation.ollama_endpoint}"
        )

        print("🚀 Generating answer...")
        answer = await generator.generate(question, passages)

        print(f"✅ Answer: {answer}")
        print(f"📏 Answer length: {len(answer)} characters")

    except Exception as exc:
        print(f"❌ Error: {exc}")
        print("🔍 Traceback:")
        traceback.print_exc()
    finally:
        await generator.cleanup()


if __name__ == "__main__":
    asyncio.run(test_ollama())
