"""Configuration management using Pydantic settings."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Database configuration.

    Backward-compatible top-level fields (host/port/collection_name) are provided
    to satisfy existing tests and callers. They mirror Milvus settings by default.
    """

    # Backward-compatible top-level fields expected by tests
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "askme_hybrid"

    class MilvusConfig(BaseSettings):
        host: str = "localhost"
        port: int = 19530
        username: str = ""
        password: str = ""
        secure: bool = False
        collection_name: str = "askme_hybrid"

    class WeaviateConfig(BaseSettings):
        url: str = "http://localhost:8081"
        api_key: str = ""
        class_name: str = "AskmeDocument"

    class QdrantConfig(BaseSettings):
        url: str = "http://localhost:6333"
        api_key: str = ""
        collection_name: str = "askme"

    milvus: MilvusConfig = MilvusConfig()
    weaviate: WeaviateConfig = WeaviateConfig()
    qdrant: QdrantConfig = QdrantConfig()

    def model_post_init(self, __context: Any) -> None:
        # Keep top-level and milvus fields in sync for backward compatibility
        # Always prefer milvus nested config over top-level defaults
        try:
            # Sync milvus config to top-level for backward compatibility
            self.host = self.milvus.host
            self.port = self.milvus.port
            self.collection_name = self.milvus.collection_name
        except Exception:
            # Be defensive: don't let post-init syncing break settings loading
            pass


class HybridConfig(BaseSettings):
    """Hybrid search configuration."""

    mode: str = "rrf"  # alpha | rrf | relative_score | ranked
    # Backward-compat: tests expect this flag
    use_rrf: bool = True
    alpha: float = 0.5
    rrf_k: int = 60
    topk: int = 50
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    enable_metadata_filter: bool = True


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""

    backend: str = "qwen3-hybrid"
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    model_name: str = "qwen3-hybrid"
    dimension: int = 1024
    max_length: int = 8192
    normalize_embeddings: bool = True
    batch_size: int = 16
    use_fp16: bool = True
    pooling_method: str = "last_token"
    query_instruction: str = ""
    passage_instruction: str = ""
    device: str = "auto"  # auto | cpu | cuda | mps

    class SparseConfig(BaseSettings):
        enabled: bool = True
        backend: str = "bge_m3"
        model: str = "BAAI/bge-m3"
        dimension: int = 1024
        batch_size: int = 4  # BGE-M3 optimal for corpus processing
        query_batch_size: int = 12  # BGE-M3 optimal for query processing
        max_length: int = 8192  # BGE-M3 official optimal
        use_fp16: bool = True
        device: str = "auto"

    sparse: SparseConfig = SparseConfig()


class RerankConfig(BaseSettings):
    """Reranking configuration."""

    local_backend: str = "qwen_local"  # qwen_local | bge_local
    local_model: str = "Qwen/Qwen3-Reranker-0.6B"
    local_enabled: bool = True
    local_batch_size: int = 16  # Qwen3-Reranker-0.6B runs comfortably at larger batches
    local_max_length: int = 1024  # Optimized length
    local_instruction: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    local_use_fp16: bool = True
    local_flash_attention: bool = False

    top_n: int = 8
    score_threshold: float = 0.0
    enable_cross_encoder: bool = True


class EnhancerConfig(BaseSettings):
    """Query enhancement configuration."""

    class HydeConfig(BaseSettings):
        enabled: bool = False
        prompt_template: str = (
            "Please write a passage to answer the question: {query}\nPassage:"
        )
        max_tokens: int = 256
        temperature: float = 0.3

    class RagFusionConfig(BaseSettings):
        enabled: bool = False
        num_queries: int = 3
        query_generation_prompt: str = ""
        fusion_method: str = "rrf"

    hyde: HydeConfig = HydeConfig()
    rag_fusion: RagFusionConfig = RagFusionConfig()


class DocumentConfig(BaseSettings):
    """Document processing configuration."""

    supported_formats: List[str] = ["pdf", "txt", "md", "html", "json", "docx"]

    class ChunkingConfig(BaseSettings):
        method: str = "semantic"
        chunk_size: int = 1000
        chunk_overlap: int = 200
        min_chunk_size: int = 100
        max_chunk_size: int = 2000

    class PreprocessingConfig(BaseSettings):
        remove_extra_whitespace: bool = True
        normalize_unicode: bool = True
        extract_metadata: bool = True
        preserve_structure: bool = True

    chunking: ChunkingConfig = ChunkingConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()


class GenerationConfig(BaseSettings):
    """LLM generation configuration."""

    # Provider: simple (local template), ollama, openai
    provider: str = "simple"
    model_name: str = "gpt-4"
    max_tokens: int = 1500
    temperature: float = 0.1
    top_p: float = 0.9

    # Ollama local settings
    ollama_model: str = "llama3.1:latest"
    ollama_endpoint: str = "http://localhost:11434"

    # OpenAI-compatible settings
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key_env: str = "OPENAI_API_KEY"  # env var name to read

    system_prompt: str = """You are a helpful assistant that answers questions based on the provided context.  # noqa: E501
Provide clean answers without inline citations - sources will be provided separately.
If you cannot find relevant information in the context, say so clearly."""

    user_prompt_template: str = """Context:  # noqa: E501
{context}

Question: {question}

Please provide a detailed answer based on the context above. Do not include citations in the answer text itself."""


class EvaluationConfig(BaseSettings):
    """Evaluation configuration."""

    class TruLensConfig(BaseSettings):
        enabled: bool = True
        metrics: List[str] = ["context_relevance", "groundedness", "answer_relevance"]
        feedback_mode: str = "with_cot_reasons"

    class RagasConfig(BaseSettings):
        enabled: bool = True
        version: str = "0.2.0"
        metrics: List[str] = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]

    class DatasetConfig(BaseSettings):
        baseline: str = "data/eval/baseline_qa.jsonl"
        custom: str = "data/eval/custom_qa.jsonl"

    class ThresholdConfig(BaseSettings):
        trulens_min: float = 0.7
        ragas_faithfulness_min: float = 0.7
        ragas_precision_min: float = 0.6
        answer_consistency_min: float = 0.9

    trulens: TruLensConfig = TruLensConfig()
    ragas: RagasConfig = RagasConfig()
    datasets: DatasetConfig = DatasetConfig()
    thresholds: ThresholdConfig = ThresholdConfig()


class APIConfig(BaseSettings):
    """API server configuration."""

    host: str = "0.0.0.0"  # nosec B104 - expose API for container networking
    port: int = 8080
    workers: int = 1
    reload: bool = False
    access_log: bool = True

    class RateLimitConfig(BaseSettings):
        enabled: bool = True
        requests_per_minute: int = 60
        burst_size: int = 10

    class CORSConfig(BaseSettings):
        allow_origins: List[str] = ["*"]
        allow_methods: List[str] = ["GET", "POST"]
        allow_headers: List[str] = ["*"]

    rate_limit: RateLimitConfig = RateLimitConfig()
    cors: CORSConfig = CORSConfig()


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    class FileConfig(BaseSettings):
        enabled: bool = True
        path: str = "logs/askme.log"
        rotation: str = "1 day"
        retention: str = "30 days"

    class StructuredConfig(BaseSettings):
        enabled: bool = True
        include_trace_id: bool = True

    file: FileConfig = FileConfig()
    structured: StructuredConfig = StructuredConfig()


class PerformanceConfig(BaseSettings):
    """Performance and monitoring configuration."""

    class IngestionConfig(BaseSettings):
        max_workers: Optional[int] = Field(
            default=None,
            description=(
                "Number of worker threads for ingestion pipeline. "
                "Defaults to CPU count when unset or <= 0."
            ),
        )

    class CacheConfig(BaseSettings):
        enabled: bool = True
        ttl_seconds: int = 3600
        max_size: int = 1000

    class BatchConfig(BaseSettings):
        embedding_batch_size: int = 16  # Balanced for Qwen3 dense processing
        sparse_batch_size: int = 4  # BGE-M3 corpus optimal
        query_batch_size: int = 12  # BGE-M3 query optimal
        rerank_batch_size: int = 16  # Qwen3-Reranker-0.6B optimal
        max_concurrent_requests: int = 8  # Reduced for memory management

    class TimeoutConfig(BaseSettings):
        embedding_timeout: int = 30
        retrieval_timeout: int = 15
        rerank_timeout: int = 30
        generation_timeout: int = 60

    ingestion: IngestionConfig = IngestionConfig()
    cache: CacheConfig = CacheConfig()
    batch: BatchConfig = BatchConfig()
    timeouts: TimeoutConfig = TimeoutConfig()


class SecurityConfig(BaseSettings):
    """Security and privacy configuration."""

    class PrivacyConfig(BaseSettings):
        log_queries: bool = False
        log_documents: bool = False
        anonymize_logs: bool = True

    class APIKeysConfig(BaseSettings):
        enabled: bool = False
        header_name: str = "X-API-Key"

    class AuditConfig(BaseSettings):
        enabled: bool = True
        log_external_calls: bool = True
        include_response_metadata: bool = True

    privacy: PrivacyConfig = PrivacyConfig()
    api_keys: APIKeysConfig = APIKeysConfig()
    audit: AuditConfig = AuditConfig()


class Settings(BaseSettings):
    """Main application settings."""

    # Vector backend selection
    vector_backend: str = Field(default="weaviate")

    # Component configurations
    database: DatabaseConfig = DatabaseConfig()
    hybrid: HybridConfig = HybridConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    rerank: RerankConfig = RerankConfig()
    enhancer: EnhancerConfig = EnhancerConfig()
    document: DocumentConfig = DocumentConfig()
    generation: GenerationConfig = GenerationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    api: APIConfig = APIConfig()
    logging: LoggingConfig = LoggingConfig()
    performance: PerformanceConfig = PerformanceConfig()
    security: SecurityConfig = SecurityConfig()

    # Environment variable overrides (kept simple for typing and tests)
    enable_ollama: bool = False
    log_level: str = "INFO"

    # Pydantic v2 style settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="ASKME_",
    )

    @classmethod
    def from_yaml(cls, config_path: str = "configs/askme.yaml") -> "Settings":
        """Load settings from YAML file with environment variable overrides."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
        else:
            config_data = {}

        # Override with environment variables manually since passing **kwargs
        # to __init__ prevents Pydantic from checking environment. Support
        # nested fields using double-underscore notation (e.g.
        # ASKME_GENERATION__OLLAMA_MODEL).
        env_prefix = "ASKME_"
        model_fields = cls.model_fields

        for env_key, env_value in os.environ.items():
            if not env_key.startswith(env_prefix):
                continue

            path = env_key[len(env_prefix) :].lower().split("__")
            if not path:
                continue

            top_key = path[0]
            if top_key not in model_fields:
                continue

            cursor = config_data
            for part in path[:-1]:
                if not isinstance(cursor, dict):
                    break
                cursor = cursor.setdefault(part, {})
            else:
                if isinstance(cursor, dict):
                    cursor[path[-1]] = env_value

        return cls(**config_data)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings.from_yaml()


# --- Test/mocking compatibility shim ---
# Python's unittest.mock.MagicMock(spec=Settings) inspects attributes on the class
# object. With Pydantic BaseSettings, field attributes are provided on instances
# but not necessarily present on the class __dict__. To make spec-based mocking in
# tests like MagicMock(spec=Settings) work with attribute access such as
# settings.rerank.local_backend, we reattach representative attributes on the
# class so they appear in dir(Settings).
try:  # be defensive: don't raise at import time in production
    _mockable_attrs = [
        "database",
        "hybrid",
        "embedding",
        "rerank",
        "enhancer",
        "document",
        "generation",
        "evaluation",
        "api",
        "logging",
        "performance",
        "security",
    ]
    for _name in _mockable_attrs:
        if not hasattr(Settings, _name):
            setattr(Settings, _name, None)
except Exception:
    pass
