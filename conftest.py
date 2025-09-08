"""
Lightweight async test support so async def tests run without external plugins.
"""

import asyncio
import inspect
from typing import Any


def pytest_pyfunc_call(pyfuncitem: Any) -> bool | None:
    """Allow running `async def` tests by driving the event loop.

    This mirrors minimal functionality of pytest-asyncio for our test needs.
    Returns True when we handled the call, allowing pytest to continue.
    """
    test_fn = pyfuncitem.obj
    if inspect.iscoroutinefunction(test_fn):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            # Pass only accepted arguments to the coroutine
            sig = inspect.signature(test_fn)
            accepted_args = {
                k: v for k, v in pyfuncitem.funcargs.items() if k in sig.parameters
            }
            loop.run_until_complete(test_fn(**accepted_args))
        finally:
            try:
                pending = asyncio.all_tasks(loop=loop)
                for task in pending:
                    task.cancel()
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            except Exception:
                pass
            loop.close()
            asyncio.set_event_loop(None)
        return True
    return None
