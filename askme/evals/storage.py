"""
Lightweight local storage for evaluation runs.

Persists JSON files under data/eval_runs/<run_id>.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

RUNS_DIR = Path("data/eval_runs")


def ensure_dir() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_run(run_id: str, payload: Dict[str, Any]) -> Path:
    ensure_dir()
    path = RUNS_DIR / f"{run_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    return path


def load_run(run_id: str) -> Optional[Dict[str, Any]]:
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        # Best-effort typing: persisted payloads are dict-like
        return cast(Dict[str, Any], data)


def list_runs(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    ensure_dir()
    files = sorted(
        RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    items: List[Dict[str, Any]] = []
    for p in files[offset : offset + limit]:
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            items.append(
                {
                    "run_id": data.get("run_id", p.stem),
                    "suite": data.get("suite", ""),
                    "status": data.get("status", "completed"),
                    "started_at": data.get("started_at", ""),
                    "total_samples": data.get("total_samples", 0),
                    "summary": data.get("summary", {}),
                }
            )
        except Exception:
            continue
    return {"runs": items, "total": len(files), "limit": limit, "offset": offset}
