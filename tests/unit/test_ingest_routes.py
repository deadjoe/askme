"""
Unit tests for askme.api.routes.ingest to raise coverage.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from fastapi import HTTPException

from askme.api.routes.ingest import (
    IngestDirectoryPayload,
    IngestFilePayload,
    IngestRequest,
    cancel_ingestion_task,
    get_ingestion_stats,
    get_ingestion_status,
    ingest_directory_endpoint,
    ingest_documents,
    ingest_file_endpoint,
    upload_and_ingest,
)
from askme.core.config import Settings
from askme.ingest.ingest_service import IngestionStats, IngestionTask, TaskStatus


class _Req:
    def __init__(self: Any) -> None:
        self.app = SimpleNamespace(state=SimpleNamespace())


@pytest.mark.asyncio
async def test_ingest_documents_validations_tmpdir(tmp_path: Path) -> None:
    settings = Settings()
    req = _Req()

    # invalid source
    with pytest.raises(HTTPException) as exc:
        await ingest_documents(IngestRequest(source="foo", path="/nope"), req, settings)
    assert exc.value.status_code == 400

    # file path not exists
    with pytest.raises(HTTPException) as exc:
        await ingest_documents(
            IngestRequest(source="file", path=str(tmp_path / "x.txt")), req, settings
        )
    assert exc.value.status_code == 404

    # dir path not exists
    with pytest.raises(HTTPException) as exc:
        await ingest_documents(
            IngestRequest(source="dir", path=str(tmp_path / "d")), req, settings
        )
    assert exc.value.status_code == 404

    # path is not a file
    d = tmp_path / "ad"
    d.mkdir()
    with pytest.raises(HTTPException) as exc:
        await ingest_documents(IngestRequest(source="file", path=str(d)), req, settings)
    assert exc.value.status_code == 400

    # path is not a directory
    f = tmp_path / "a.txt"
    f.write_text("hi")
    with pytest.raises(HTTPException) as exc:
        await ingest_documents(IngestRequest(source="dir", path=str(f)), req, settings)
    assert exc.value.status_code == 400

    # URL not implemented
    with pytest.raises(HTTPException) as exc:
        await ingest_documents(
            IngestRequest(source="url", path="http://x"), req, settings
        )
    assert exc.value.status_code == 501


@pytest.mark.asyncio
async def test_ingest_documents_calls_service(tmp_path: Path) -> None:
    settings = Settings()
    req = _Req()

    class _Svc:
        async def ingest_file(self: Any, **kwargs: Any) -> str:
            return "tid-file"

        async def ingest_directory(self: Any, **kwargs: Any) -> str:
            return "tid-dir"

    req.app.state.ingestion_service = _Svc()

    # file happy path
    f = tmp_path / "a.txt"
    f.write_text("hi")
    out = await ingest_documents(
        IngestRequest(source="file", path=str(f)), req, settings
    )
    assert out.task_id == "tid-file"

    # dir happy path
    d = tmp_path / "ad"
    d.mkdir()
    out = await ingest_documents(
        IngestRequest(source="dir", path=str(d)), req, settings
    )
    assert out.task_id == "tid-dir"


@pytest.mark.asyncio
async def test_ingest_file_and_directory_endpoints_call_service(
    tmp_path: Path,
) -> None:
    settings = Settings()
    req = _Req()

    class _Svc:
        async def ingest_file(self: Any, **kwargs: Any) -> str:
            return "fid"

        async def ingest_directory(self: Any, **kwargs: Any) -> str:
            return "did"

    req.app.state.ingestion_service = _Svc()

    f_out = await ingest_file_endpoint(
        IngestFilePayload(file_path=str(tmp_path / "f.txt")), req, settings
    )
    assert f_out.task_id == "fid"

    d_out = await ingest_directory_endpoint(
        IngestDirectoryPayload(directory_path=str(tmp_path)), req, settings
    )
    assert d_out.task_id == "did"


@pytest.mark.asyncio
async def test_upload_and_ingest_validation(tmp_path: Path) -> None:
    settings = Settings()

    class _Upload:
        def __init__(self: Any, name: str) -> Any:
            self.filename = name

    # unsupported extension
    bad = [_Upload("bad.exe")]
    with pytest.raises(HTTPException) as exc:
        await upload_and_ingest(bad, tags=None, overwrite=False, settings=settings)
    assert exc.value.status_code == 400

    # supported extensions (derived from settings)
    ok = [_Upload("a.pdf"), _Upload("b.txt")]
    resp = await upload_and_ingest(ok, tags="t1,t2", overwrite=True, settings=settings)
    assert resp.document_count == 2 and resp.status == "queued"


@pytest.mark.asyncio
async def test_get_ingestion_status_and_stats() -> None:
    settings = Settings()
    req = _Req()

    # service unavailable
    with pytest.raises(HTTPException) as exc:
        await get_ingestion_status("tid", req, settings)
    assert exc.value.status_code == 503

    # attach service
    class _Svc:
        def __init__(self: Any) -> None:
            now = datetime.utcnow()
            self._task = IngestionTask(
                task_id="tid",
                source_type="file",
                source_path="/tmp/x",
                status=TaskStatus.PROCESSING,
                created_at=now - timedelta(seconds=5),
                started_at=now - timedelta(seconds=5),
                completed_at=None,
                total_files=2,
                processed_files=1,
                total_chunks=10,
                processed_chunks=4,
            )

        async def get_task_status(self: Any, tid: str) -> Optional[IngestionTask]:
            return self._task if tid == "tid" else None

        async def get_ingestion_stats(self: Any) -> IngestionStats:
            return IngestionStats(
                total_documents=3,
                total_chunks=15,
                total_size_bytes=0,
                processing_time_seconds=12.3,
                files_by_type={},
                chunks_per_document=5.0,
                errors=["e1"],
            )

    svc = _Svc()
    req.app.state.ingestion_service = svc

    status = await get_ingestion_status("tid", req, settings)
    assert status.status in {"processing", "completed", "queued", "failed", "cancelled"}
    assert status.progress > 0

    # missing task
    with pytest.raises(HTTPException) as exc:
        await get_ingestion_status("missing", req, settings)
    assert exc.value.status_code == 404

    stats = await get_ingestion_stats(req, settings)
    assert stats["total_documents"] == 3


@pytest.mark.asyncio
async def test_cancel_endpoint() -> None:
    out = await cancel_ingestion_task("tid", Settings())
    assert out["status"] == "cancelled"
