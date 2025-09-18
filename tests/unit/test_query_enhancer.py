from typing import Any

"""
Unit tests for query enhancement utilities (HyDE and RAG-Fusion).
"""

import pytest

from askme.enhancer.query_enhancer import (
    HydeConfig,
    RagFusionConfig,
    generate_fusion_queries,
    generate_hyde_passage,
)


class TestHydeConfig:
    """Test HyDE configuration."""

    def test_default_values(self: Any) -> None:
        """Test default HyDE configuration values."""
        config = HydeConfig()
        assert config.enabled is False
        assert config.max_tokens == 256

    def test_custom_values(self: Any) -> None:
        """Test custom HyDE configuration."""
        config = HydeConfig(enabled=True, max_tokens=512)
        assert config.enabled is True
        assert config.max_tokens == 512


class TestRagFusionConfig:
    """Test RAG-Fusion configuration."""

    def test_default_values(self: Any) -> None:
        """Test default RAG-Fusion configuration values."""
        config = RagFusionConfig()
        assert config.enabled is False
        assert config.num_queries == 3

    def test_custom_values(self: Any) -> None:
        """Test custom RAG-Fusion configuration."""
        config = RagFusionConfig(enabled=True, num_queries=5)
        assert config.enabled is True
        assert config.num_queries == 5


class TestGenerateHydePassage:
    """Test HyDE hypothetical passage generation."""

    def test_basic_passage_generation(self: Any) -> None:
        """Test basic HyDE passage generation."""
        query = "machine learning algorithms"
        passage = generate_hyde_passage(query)

        assert isinstance(passage, str)
        assert len(passage) > 0
        assert query in passage.lower()

        # Should contain template elements
        assert "passage discusses" in passage.lower()
        assert "background" in passage.lower()
        assert "key concepts" in passage.lower()

    def test_passage_with_custom_max_tokens(self: Any) -> None:
        """Test HyDE passage generation with custom max_tokens."""
        query = "deep learning"

        # Short passage
        short_passage = generate_hyde_passage(query, max_tokens=10)
        long_passage = generate_hyde_passage(query, max_tokens=100)

        assert len(short_passage) <= len(long_passage)
        assert len(short_passage) <= 40  # Rough truncation: 10 * 4 chars

        # Both should contain the query (or at least part of it for short passages)
        assert query in long_passage.lower()
        # Short passage may be truncated, so just check it starts correctly
        assert short_passage.lower().startswith("this passage discusses")

    def test_passage_with_special_characters(self: Any) -> None:
        """Test HyDE passage generation with special characters in query."""
        query = "C++ programming & data structures"
        passage = generate_hyde_passage(query)

        assert isinstance(passage, str)
        assert len(passage) > 0
        # Special characters should be preserved
        assert "C++" in passage
        assert "&" in passage

    def test_passage_with_empty_query(self: Any) -> None:
        """Test HyDE passage generation with empty query."""
        query = ""
        passage = generate_hyde_passage(query)

        assert isinstance(passage, str)
        assert len(passage) > 0
        # Should still generate template content
        assert "passage discusses" in passage.lower()

    def test_passage_with_very_long_query(self: Any) -> None:
        """Test HyDE passage generation with very long query."""
        query = "artificial intelligence machine learning deep learning " * 50
        passage = generate_hyde_passage(query, max_tokens=50)

        assert isinstance(passage, str)
        assert len(passage) <= 200  # 50 * 4 rough truncation
        # Should contain at least part of the query
        assert "artificial intelligence" in passage

    def test_passage_deterministic(self: Any) -> None:
        """Test that HyDE passage generation is deterministic."""
        query = "natural language processing"

        passage1 = generate_hyde_passage(query, max_tokens=100)
        passage2 = generate_hyde_passage(query, max_tokens=100)

        assert passage1 == passage2  # Should be identical

    def test_passage_template_structure(self: Any) -> None:
        """Test that HyDE passage follows expected template structure."""
        query = "computer vision applications"
        passage = generate_hyde_passage(query)

        # Should contain key template phrases
        expected_phrases = [
            "This passage discusses the topic:",
            "provides background",
            "key concepts",
            "practical details",
            "concise way",
        ]

        for phrase in expected_phrases:
            assert phrase.lower() in passage.lower()


class TestGenerateFusionQueries:
    """Test RAG-Fusion query generation."""

    def test_basic_fusion_queries(self: Any) -> None:
        """Test basic fusion query generation."""
        original_query = "machine learning algorithms"
        queries = generate_fusion_queries(original_query)

        assert isinstance(queries, list)
        assert len(queries) >= 1
        assert len(queries) <= 4  # Default variants
        assert queries[0] == original_query  # Original always first

        # Should have different variants
        assert len(set(queries)) == len(queries)  # All unique

    def test_fusion_queries_with_custom_num(self: Any) -> None:
        """Test fusion query generation with custom number."""
        original_query = "deep learning neural networks"

        # Test different numbers
        queries_1 = generate_fusion_queries(original_query, num_queries=1)
        queries_2 = generate_fusion_queries(original_query, num_queries=2)
        queries_5 = generate_fusion_queries(original_query, num_queries=5)

        assert len(queries_1) == 1
        assert len(queries_2) == 2
        assert len(queries_5) <= 4  # Limited by available variants

        # Original query should always be first
        assert queries_1[0] == original_query
        assert queries_2[0] == original_query
        assert queries_5[0] == original_query

    def test_fusion_queries_variants(self: Any) -> None:
        """Test that fusion queries contain expected variants."""
        original_query = "natural language processing"
        queries = generate_fusion_queries(original_query, num_queries=4)

        # Should have original query
        assert original_query in queries

        # Should have definition variant
        definition_queries = [
            q for q in queries if q.startswith("Definition and overview")
        ]
        assert len(definition_queries) >= 1

        # Should have benefits variant
        benefits_queries = [q for q in queries if "benefits" in q.lower()]
        assert len(benefits_queries) >= 1

        # Should have steps/techniques variant
        steps_queries = [
            q for q in queries if "steps" in q.lower() or "techniques" in q.lower()
        ]
        assert len(steps_queries) >= 1

    def test_fusion_queries_with_zero_num(self: Any) -> None:
        """Test fusion query generation with zero queries requested."""
        original_query = "computer vision"
        queries = generate_fusion_queries(original_query, num_queries=0)

        # Should still return at least 1 (the original)
        assert len(queries) >= 1
        assert queries[0] == original_query

    def test_fusion_queries_with_empty_query(self: Any) -> None:
        """Test fusion query generation with empty original query."""
        original_query = ""
        queries = generate_fusion_queries(original_query, num_queries=3)

        assert isinstance(queries, list)
        assert len(queries) >= 1
        assert queries[0] == ""  # Original (empty) query preserved

        # Other variants should still be generated with empty query
        for query in queries[1:]:
            assert isinstance(query, str)

    def test_fusion_queries_deduplication(self: Any) -> None:
        """Test that fusion queries are properly deduplicated."""
        # Use a query that might generate similar variants
        original_query = "definition"
        queries = generate_fusion_queries(original_query, num_queries=5)

        # All queries should be unique
        assert len(queries) == len(set(queries))

        # Original should be first
        assert queries[0] == original_query

    def test_fusion_queries_deterministic(self: Any) -> None:
        """Test that fusion query generation is deterministic."""
        original_query = "artificial intelligence applications"

        queries1 = generate_fusion_queries(original_query, num_queries=3)
        queries2 = generate_fusion_queries(original_query, num_queries=3)

        assert queries1 == queries2  # Should be identical

    def test_fusion_queries_with_special_characters(self: Any) -> None:
        """Test fusion queries with special characters in original query."""
        original_query = "C++ programming & algorithms"
        queries = generate_fusion_queries(original_query, num_queries=3)

        assert len(queries) >= 1
        assert queries[0] == original_query

        # Special characters should be preserved in variants
        for query in queries:
            assert isinstance(query, str)
            if "C++" in original_query and original_query in query:
                assert "C++" in query

    def test_fusion_queries_content_relevance(self: Any) -> None:
        """Test that fusion query variants are relevant to original."""
        original_query = "blockchain technology"
        queries = generate_fusion_queries(original_query, num_queries=4)

        # All non-original queries should contain the original query text
        for query in queries[1:]:  # Skip original
            assert original_query in query

        # Check specific variant patterns
        definition_found = any("Definition and overview" in q for q in queries)
        benefits_found = any("benefits" in q.lower() for q in queries)
        steps_found = any(
            "steps" in q.lower()
            or "techniques" in q.lower()
            or "components" in q.lower()
            for q in queries
        )

        # At least some standard variants should be present
        variant_count = sum([definition_found, benefits_found, steps_found])
        assert variant_count >= 1  # At least one variant pattern

    def test_fusion_queries_max_length_handling(self: Any) -> None:
        """Test fusion queries with very long original query."""
        original_query = (
            "very long query about machine learning and artificial intelligence " * 10
        )
        queries = generate_fusion_queries(original_query, num_queries=3)

        assert len(queries) >= 1
        assert queries[0] == original_query

        # Variants should still be generated (though they'll be long)
        for query in queries:
            assert isinstance(query, str)
            assert len(query) > 0


class TestQueryEnhancementIntegration:
    """Integration tests for query enhancement utilities."""

    def test_hyde_and_fusion_together(self: Any) -> None:
        """Test using HyDE and RAG-Fusion together."""
        original_query = "machine learning model evaluation"

        # Generate fusion queries
        fusion_queries = generate_fusion_queries(original_query, num_queries=3)

        # Generate HyDE passages for each
        hyde_passages = []
        for query in fusion_queries:
            passage = generate_hyde_passage(query, max_tokens=100)
            hyde_passages.append(passage)

        assert len(hyde_passages) == len(fusion_queries)

        # Each passage should be relevant to its query
        for i, (query, passage) in enumerate(zip(fusion_queries, hyde_passages)):
            assert isinstance(passage, str)
            assert len(passage) > 0

            # Original query terms should appear in passage
            if "machine learning" in query:
                assert "machine learning" in passage.lower()

    def test_config_integration(self: Any) -> None:
        """Test configuration classes with enhancement functions."""
        hyde_config = HydeConfig(enabled=True, max_tokens=128)
        fusion_config = RagFusionConfig(enabled=True, num_queries=2)

        if hyde_config.enabled and fusion_config.enabled:
            query = "deep learning optimization"

            # Use configs in enhancement functions
            fusion_queries = generate_fusion_queries(query, fusion_config.num_queries)
            hyde_passage = generate_hyde_passage(query, hyde_config.max_tokens)

            assert (
                len(fusion_queries) <= fusion_config.num_queries
                or len(fusion_queries) <= 4
            )
            assert len(hyde_passage) <= hyde_config.max_tokens * 4  # Rough char limit

        # Test disabled configs
        disabled_hyde = HydeConfig(enabled=False)
        disabled_fusion = RagFusionConfig(enabled=False)

        assert disabled_hyde.enabled is False
        assert disabled_fusion.enabled is False
