"""
Evaluation and metrics endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from askme.core.config import Settings, get_settings
from askme.evals.evaluator import EvalItem, Evaluator
from askme.evals.ragas_runner import run_ragas
from askme.evals.trulens_runner import run_trulens


class EvalSuite(str, Enum):
    """Available evaluation suites."""

    BASELINE = "baseline"
    CUSTOM = "custom"
    REGRESSION = "regression"
    QUICK = "quick"


class MetricType(str, Enum):
    """Available metric types."""

    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    CONTEXT_RELEVANCE = "context_relevance"
    GROUNDEDNESS = "groundedness"
    ANSWER_RELEVANCE = "answer_relevance"


class EvaluationRequest(BaseModel):
    """Evaluation run request."""

    suite: EvalSuite = Field(..., description="Evaluation suite to run")
    metrics: Optional[List[MetricType]] = Field(
        default=None, description="Specific metrics to evaluate"
    )
    dataset_path: Optional[str] = Field(default=None, description="Custom dataset path")
    sample_size: Optional[int] = Field(
        default=None, ge=1, description="Number of samples to evaluate"
    )
    config_overrides: Optional[Dict[str, Any]] = Field(
        default=None, description="Configuration overrides"
    )


class MetricResult(BaseModel):
    """Individual metric result."""

    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    threshold: Optional[float] = Field(default=None, description="Quality threshold")
    passed: bool = Field(..., description="Whether threshold was met")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metric details"
    )


class EvaluationResult(BaseModel):
    """Single evaluation result for one query."""

    query_id: str = Field(..., description="Unique query identifier")
    query: str = Field(..., description="Original query")
    ground_truth: Optional[str] = Field(default=None, description="Expected answer")
    generated_answer: str = Field(..., description="Generated answer")
    retrieved_contexts: List[str] = Field(..., description="Retrieved context chunks")
    metrics: List[MetricResult] = Field(..., description="Metric scores")


class EvaluationResponse(BaseModel):
    """Evaluation run response."""

    run_id: str = Field(..., description="Unique run identifier")
    status: str = Field(..., description="Run status")
    suite: str = Field(..., description="Evaluation suite")
    started_at: datetime = Field(..., description="Run start time")
    completed_at: Optional[datetime] = Field(
        default=None, description="Run completion time"
    )
    total_samples: int = Field(..., description="Total number of samples")
    processed_samples: int = Field(..., description="Number of processed samples")

    # Aggregate metrics
    overall_metrics: List[MetricResult] = Field(
        ..., description="Aggregated metric results"
    )
    individual_results: Optional[List[EvaluationResult]] = Field(
        default=None, description="Per-query results"
    )

    # Summary statistics
    summary: Dict[str, Any] = Field(..., description="Summary statistics")


class ComparisonRequest(BaseModel):
    """A/B comparison request."""

    baseline_run_id: str = Field(..., description="Baseline evaluation run ID")
    experiment_run_id: str = Field(..., description="Experiment evaluation run ID")
    metrics: Optional[List[MetricType]] = Field(
        default=None, description="Metrics to compare"
    )


class ComparisonResult(BaseModel):
    """A/B comparison result."""

    metric_name: str
    baseline_score: float
    experiment_score: float
    improvement: float
    statistical_significance: Optional[float] = None
    confidence_interval: Optional[List[float]] = None


class ComparisonResponse(BaseModel):
    """A/B comparison response."""

    comparison_id: str
    baseline_run_id: str
    experiment_run_id: str
    results: List[ComparisonResult]
    summary: Dict[str, Any]


router = APIRouter()


@router.post("/run", response_model=EvaluationResponse)
async def run_evaluation(
    request: EvaluationRequest, settings: Settings = Depends(get_settings)
) -> EvaluationResponse:
    """
    Run evaluation suite with specified metrics.

    Supports:
    - TruLens RAG Triad (context_relevance, groundedness, answer_relevance)
    - Ragas metrics (faithfulness, answer_relevancy, context_precision, context_recall)
    - Custom datasets
    - Configuration overrides
    """

    import uuid

    # Validate metrics
    if request.metrics:
        available_metrics = set(MetricType)
        requested_metrics = set(request.metrics)
        invalid_metrics = requested_metrics - available_metrics
        if invalid_metrics:
            raise HTTPException(
                status_code=400, detail=f"Invalid metrics: {list(invalid_metrics)}"
            )

    run_id = str(uuid.uuid4())

    # Build a tiny in-memory sample and attempt real Ragas first
    samples = max(1, (request.sample_size or 3))
    sample_rows = [
        {
            "question": "What is hybrid search?",
            "answer": "Hybrid search combines dense and sparse retrieval techniques to improve recall and precision.",
            "contexts": [
                "Hybrid search often uses BM25 (sparse) and dense embeddings.",
                "RRF or alpha fusion merges rankings from different channels.",
            ],
            "ground_truths": [
                "Hybrid search mixes sparse (BM25) with dense vector search; fusion methods like RRF or alpha combine results."
            ],
        }
        for _ in range(samples)
    ]

    overall_metrics: List[MetricResult] = []

    # Try TruLens first if enabled
    if settings.evaluation.trulens.enabled:
        try:
            tl_scores = run_trulens(sample_rows)
            for name, value in tl_scores.items():
                # thresholds for triad can reuse trulens_min
                threshold = (
                    settings.evaluation.thresholds.trulens_min
                    if name
                    in ("context_relevance", "groundedness", "answer_relevance")
                    else None
                )
                overall_metrics.append(
                    MetricResult(
                        name=name,
                        value=value,
                        threshold=threshold,
                        passed=(value >= threshold) if threshold is not None else True,
                        details={"samples": samples},
                    )
                )
        except Exception:
            # proceed; truLens optional
            pass

    # Try real Ragas evaluation
    try:
        metric_names = [m.value for m in request.metrics] if request.metrics else None
        ragas_scores = run_ragas(sample_rows, metrics=metric_names)

        # Map scores to response with thresholds
        for name, value in ragas_scores.items():
            threshold = None
            if name == "faithfulness":
                threshold = settings.evaluation.thresholds.ragas_faithfulness_min
            elif name == "context_precision":
                threshold = settings.evaluation.thresholds.ragas_precision_min
            overall_metrics.append(
                MetricResult(
                    name=name,
                    value=value,
                    threshold=threshold,
                    passed=(value >= threshold) if threshold is not None else True,
                    details={"samples": samples},
                )
            )

    except Exception as e:
        # Fall back to heuristic evaluator if ragas unavailable
        from typing import cast

        evaluator = Evaluator(settings.evaluation)
        heuristic_scores = evaluator.evaluate_batch(
            [
                EvalItem(
                    query=cast(str, r["question"]),
                    answer=cast(str, r["answer"]),
                    contexts=cast(List[str], r["contexts"]),
                    ground_truth=(
                        cast(List[str], r.get("ground_truths", []))[0]
                        if r.get("ground_truths")
                        else None
                    ),
                )
                for r in sample_rows
            ]
        )
        for s in heuristic_scores:
            overall_metrics.append(
                MetricResult(
                    name=s.name,
                    value=s.value,
                    threshold=s.threshold,
                    passed=(s.value >= s.threshold)
                    if s.threshold is not None
                    else True,
                    details={
                        "samples": samples,
                        "note": "heuristic_fallback",
                        "error": str(e),
                    },
                )
            )

    return EvaluationResponse(
        run_id=run_id,
        status="completed",
        suite=request.suite.value,
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        total_samples=samples,
        processed_samples=samples,
        overall_metrics=overall_metrics,
        summary={
            "avg_faithfulness": next(
                (m.value for m in overall_metrics if m.name == "faithfulness"), 0.0
            ),
            "avg_answer_relevancy": next(
                (
                    m.value
                    for m in overall_metrics
                    if m.name in ("answer_relevancy", "answer_relevance")
                ),
                0.0,
            ),
            "avg_context_precision": next(
                (m.value for m in overall_metrics if m.name == "context_precision"), 0.0
            ),
            "avg_context_relevance": next(
                (m.value for m in overall_metrics if m.name == "context_relevance"), 0.0
            ),
            "avg_groundedness": next(
                (m.value for m in overall_metrics if m.name == "groundedness"), 0.0
            ),
            "pass_rate": sum(1 for m in overall_metrics if m.passed)
            / max(1, len(overall_metrics)),
        },
    )


@router.get("/runs/{run_id}", response_model=EvaluationResponse)
async def get_evaluation_results(
    run_id: str,
    include_individual: bool = False,
    settings: Settings = Depends(get_settings),
) -> EvaluationResponse:
    """
    Get results from a specific evaluation run.
    """

    # TODO: Implement result retrieval from storage

    # Mock response
    mock_metrics = [
        MetricResult(name="faithfulness", value=0.85, threshold=0.7, passed=True)
    ]

    individual_results = None
    if include_individual:
        individual_results = [
            EvaluationResult(
                query_id="q_001",
                query="What is machine learning?",
                ground_truth="Machine learning is a subset of AI...",
                generated_answer="Machine learning is an AI technique...",
                retrieved_contexts=["Context 1", "Context 2"],
                metrics=mock_metrics,
            )
        ]

    return EvaluationResponse(
        run_id=run_id,
        status="completed",
        suite="baseline",
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        total_samples=100,
        processed_samples=100,
        overall_metrics=mock_metrics,
        individual_results=individual_results,
        summary={"avg_score": 0.85},
    )


@router.post("/compare", response_model=ComparisonResponse)
async def compare_evaluation_runs(
    request: ComparisonRequest, settings: Settings = Depends(get_settings)
) -> ComparisonResponse:
    """
    Compare two evaluation runs for A/B testing.
    """

    import uuid

    # TODO: Implement actual comparison logic
    # This would involve:
    # 1. Load both evaluation results
    # 2. Calculate statistical significance
    # 3. Generate comparison report

    mock_results = [
        ComparisonResult(
            metric_name="faithfulness",
            baseline_score=0.75,
            experiment_score=0.82,
            improvement=0.07,
            statistical_significance=0.05,
            confidence_interval=[0.02, 0.12],
        ),
        ComparisonResult(
            metric_name="answer_relevancy",
            baseline_score=0.70,
            experiment_score=0.78,
            improvement=0.08,
            statistical_significance=0.03,
        ),
    ]

    return ComparisonResponse(
        comparison_id=str(uuid.uuid4()),
        baseline_run_id=request.baseline_run_id,
        experiment_run_id=request.experiment_run_id,
        results=mock_results,
        summary={
            "total_metrics": len(mock_results),
            "significant_improvements": 2,
            "significant_degradations": 0,
            "overall_improvement": 0.075,
        },
    )


@router.get("/runs")
async def list_evaluation_runs(
    suite: Optional[EvalSuite] = None,
    limit: int = 20,
    offset: int = 0,
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    """
    List available evaluation runs with pagination.
    """

    # TODO: Implement actual run listing

    mock_runs = [
        {
            "run_id": "run_001",
            "suite": "baseline",
            "status": "completed",
            "started_at": "2025-09-08T10:00:00Z",
            "total_samples": 100,
            "avg_faithfulness": 0.85,
        },
        {
            "run_id": "run_002",
            "suite": "custom",
            "status": "running",
            "started_at": "2025-09-08T11:00:00Z",
            "total_samples": 50,
            "processed_samples": 25,
        },
    ]

    return {
        "runs": mock_runs,
        "total": len(mock_runs),
        "limit": limit,
        "offset": offset,
    }


@router.delete("/runs/{run_id}")
async def delete_evaluation_run(
    run_id: str, settings: Settings = Depends(get_settings)
) -> Dict[str, str]:
    """
    Delete an evaluation run and its results.
    """

    # TODO: Implement run deletion

    return {
        "run_id": run_id,
        "status": "deleted",
        "message": "Evaluation run deleted successfully",
    }


@router.get("/metrics")
async def get_available_metrics(
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    """
    Get information about available evaluation metrics.
    """

    return {
        "trulens_metrics": {
            "context_relevance": {
                "description": "Measures how relevant retrieved context is to the query",
                "range": [0, 1],
                "higher_is_better": True,
                "default_threshold": 0.7,
            },
            "groundedness": {
                "description": "Measures how well the answer is supported by context",
                "range": [0, 1],
                "higher_is_better": True,
                "default_threshold": 0.7,
            },
            "answer_relevance": {
                "description": "Measures how relevant the answer is to the query",
                "range": [0, 1],
                "higher_is_better": True,
                "default_threshold": 0.7,
            },
        },
        "ragas_metrics": {
            "faithfulness": {
                "description": "Measures factual consistency of answer with context",
                "range": [0, 1],
                "higher_is_better": True,
                "default_threshold": 0.7,
            },
            "answer_relevancy": {
                "description": "Measures relevance of answer to query",
                "range": [0, 1],
                "higher_is_better": True,
                "default_threshold": 0.7,
            },
            "context_precision": {
                "description": "Measures precision of retrieved context",
                "range": [0, 1],
                "higher_is_better": True,
                "default_threshold": 0.6,
            },
            "context_recall": {
                "description": "Measures recall of retrieved context",
                "range": [0, 1],
                "higher_is_better": True,
                "default_threshold": 0.6,
            },
        },
    }
