"""
uvloop compatibility utilities for evaluation frameworks.

This module provides utilities to handle the conflict between uvloop (used by uvicorn)
and nest_asyncio (required by TruLens and Ragas).
"""

import asyncio
import sys
from typing import Any, Callable, Dict, Optional, TypeVar, Union, cast

T = TypeVar("T")


def safe_import_eval_framework(import_func: Callable[[], T]) -> Optional[T]:
    """
    Safely import evaluation frameworks that conflict with uvloop.

    This function temporarily switches to a standard asyncio loop,
    imports the framework, then restores the original loop.
    """
    # Save current loop and policy
    original_policy = None
    current_loop = None

    try:
        # Get current event loop if any
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        # Check if we're in a uvloop environment
        if current_loop and "uvloop" in str(type(current_loop)):
            print("DEBUG: Detected uvloop, switching to standard asyncio for import")

            # Save the current policy
            original_policy = asyncio.get_event_loop_policy()

            # Switch to standard asyncio policy temporarily
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

            # The import will happen in the new policy context
            result = import_func()

            print("DEBUG: Framework import successful with standard asyncio")
            return result
        else:
            # Not in uvloop, safe to import directly
            print("DEBUG: Not in uvloop environment, importing directly")
            return import_func()

    except Exception as e:
        print(f"DEBUG: Framework import failed even with policy switch: {e}")
        return None

    finally:
        # Restore original policy if we changed it
        if original_policy is not None:
            try:
                asyncio.set_event_loop_policy(original_policy)
                print("DEBUG: Restored original event loop policy")
            except Exception as e:
                print(f"DEBUG: Failed to restore original policy: {e}")


def run_in_thread_with_new_loop(func: Callable[[], T]) -> T:
    """
    Run a function in a new thread with its own event loop.

    This is useful for evaluation frameworks that need their own async context.
    """
    import concurrent.futures
    import threading

    result_container: Dict[str, Union[T, Exception, None]] = {
        "result": None,
        "exception": None,
    }

    def thread_target() -> None:
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                result = func()
                result_container["result"] = result
            finally:
                loop.close()

        except Exception as exc:
            result_container["exception"] = exc

    thread = threading.Thread(target=thread_target)
    thread.start()
    thread.join()

    if result_container["exception"] is not None:
        raise cast(Exception, result_container["exception"])

    return cast(T, result_container["result"])


def patch_nest_asyncio_import() -> None:
    """
    Monkey patch nest_asyncio to be compatible with uvloop.

    This prevents nest_asyncio from trying to patch uvloop when imported.
    """
    try:
        import nest_asyncio

        # Store original apply function
        original_apply = nest_asyncio.apply

        def safe_apply(
            loop: Optional[asyncio.AbstractEventLoop] = None,
        ) -> None:
            """Safe version of nest_asyncio.apply that handles uvloop."""
            try:
                current_loop = loop or asyncio.get_event_loop()
                if "uvloop" in str(type(current_loop)):
                    print("DEBUG: Skipping nest_asyncio.apply for uvloop")
                    return None
                original_apply(loop)
                return None
            except Exception as e:
                print(f"DEBUG: nest_asyncio.apply failed: {e}")
                # Don't raise, just skip the patch

        # Replace the apply function
        nest_asyncio.apply = safe_apply
        print("DEBUG: Patched nest_asyncio.apply to handle uvloop")

    except ImportError:
        print("DEBUG: nest_asyncio not available, no patching needed")
    except Exception as e:
        print(f"DEBUG: Failed to patch nest_asyncio: {e}")
