#!/usr/bin/env python3
"""Test helper for checking the API generator wiring."""
import asyncio
from typing import Any, Dict

import httpx


async def test_api_generator() -> None:
    print("🔍 Testing API generator status...")

    # Test a simple endpoint that should show internal state
    async with httpx.AsyncClient() as client:
        try:
            # Test mock query first
            print("📝 Testing with mock data...")
            response = await client.post(
                "http://localhost:8080/query/",
                json={
                    "q": "test query",
                    "topk": 1,
                    "max_passages": 1,
                    "include_debug": True,
                },
                timeout=30.0,
            )

            data: Dict[str, Any] = response.json()
            print(f"✅ Status: {response.status_code}")
            print(f"📄 Response keys: {list(data.keys())}")
            print(f"🎯 Answer: '{data.get('answer', '')}'")
            print(f"📏 Answer length: {len(data.get('answer', ''))}")

            if response.status_code != 200:
                print(f"❌ Error response: {data}")

        except Exception as exc:
            print(f"❌ Request failed: {exc}")


if __name__ == "__main__":
    asyncio.run(test_api_generator())
