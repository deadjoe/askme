"""
Configuration management using Pydantic settings.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml

from pydantic import Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    
    class MilvusConfig(BaseSettings):
        host: str = "localhost"
        port: int = 19530
        username: str = ""
        password: str = ""
        secure: bool = False
        collection_name: str = "askme_hybrid"
    
    class WeaviateConfig(BaseSettings):
        url: str = "http://localhost:8080"
        api_key: str = ""
        class_name: str = "AskmeDocument"
    
    class QdrantConfig(BaseSettings):
        url: str = "http://localhost:6333"
        api_key: str = ""
        collection_name: str = "askme"
    
    milvus: MilvusConfig = MilvusConfig()
    weaviate: WeaviateConfig = WeaviateConfig()
    qdrant: QdrantConfig = QdrantConfig()


class HybridConfig(BaseSettings):
    """Hybrid search configuration."""
    mode: str = "rrf"  # alpha | rrf | relative_score | ranked
    alpha: float = 0.5
    rrf_k: int = 60
    topk: int = 50
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    enable_metadata_filter: bool = True


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""
    model: str = "BAAI/bge-m3"
    model_name: str = "bge-m3"
    dimension: int = 1024
    max_length: int = 8192
    normalize_embeddings: bool = True
    batch_size: int = 32
    use_fp16: bool = True
    pooling_method: str = "cls"
    query_instruction: str = ""
    passage_instruction: str = ""


class RerankConfig(BaseSettings):
    """Reranking configuration."""
    local_model: str = "BAAI/bge-reranker-v2.5-gemma2-lightweight"
    local_enabled: bool = True
    local_batch_size: int = 16
    local_max_length: int = 1024
    
    cohere_enabled: bool = False
    cohere_model: str = "rerank-3.5-turbo"
    cohere_max_chunks_per_doc: int = 10
    cohere_return_documents: bool = True
    
    top_n: int = 8
    score_threshold: float = 0.0
    enable_cross_encoder: bool = True


class EnhancerConfig(BaseSettings):
    """Query enhancement configuration."""
    
    class HydeConfig(BaseSettings):
        enabled: bool = False
        prompt_template: str = "Please write a passage to answer the question: {query}\nPassage:"
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
    model_name: str = "gpt-4"
    max_tokens: int = 1500
    temperature: float = 0.1
    top_p: float = 0.9
    
    system_prompt: str = """You are a helpful assistant that answers questions based on the provided context.
Always cite your sources using the provided document references.
If you cannot find relevant information in the context, say so clearly."""
    
    user_prompt_template: str = """Context:
{context}

Question: {question}

Please provide a detailed answer based on the context above. Include citations in the format [Doc ID: title]."""


class EvaluationConfig(BaseSettings):
    """Evaluation configuration."""
    
    class TruLensConfig(BaseSettings):
        enabled: bool = True
        metrics: List[str] = ["context_relevance", "groundedness", "answer_relevance"]
        feedback_mode: str = "with_cot_reasons"
    
    class RagasConfig(BaseSettings):
        enabled: bool = True
        version: str = "0.2.0"
        metrics: List[str] = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    
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
    host: str = "0.0.0.0"
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
    
    class CacheConfig(BaseSettings):
        enabled: bool = True
        ttl_seconds: int = 3600
        max_size: int = 1000
    
    class BatchConfig(BaseSettings):
        embedding_batch_size: int = 32
        rerank_batch_size: int = 16
        max_concurrent_requests: int = 10
    
    class TimeoutConfig(BaseSettings):
        embedding_timeout: int = 30
        retrieval_timeout: int = 15
        rerank_timeout: int = 30
        generation_timeout: int = 60
    
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
    vector_backend: str = Field(default="milvus", env="ASKME_VECTOR_BACKEND")
    
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
    
    # Environment variable overrides
    enable_cohere: bool = Field(default=False, env="ASKME_ENABLE_COHERE")
    log_level: str = Field(default="INFO", env="ASKME_LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @classmethod
    def from_yaml(cls, config_path: str = "configs/askme.yaml") -> "Settings":
        """Load settings from YAML file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            return cls(**config_data)
        return cls()


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings.from_yaml()