from typing import Any

"""
Unit tests for TruLens evaluation runner.
"""

from unittest.mock import MagicMock, patch

import pytest

from askme.evals.trulens_runner import run_trulens


class TestTruLensRunner:
    """Test TruLens evaluation runner."""

    def test_trulens_import_error(self: Any) -> None:
        """Test RuntimeError when TruLens is not available."""
        # Block the import by making it raise an exception
        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'trulens_eval'"),
        ):
            with pytest.raises(RuntimeError, match="TruLens not available"):
                run_trulens([])

    def test_trulens_basic_execution(self: Any) -> None:
        """Test basic TruLens execution with mocked dependencies."""
        # Mock the imported modules and classes
        mock_modules = {
            "trulens_eval": MagicMock(),
            "trulens_eval.feedback": MagicMock(),
            "trulens_eval.feedback.provider": MagicMock(),
        }

        # Mock individual classes
        mock_tru = MagicMock()
        mock_feedback = MagicMock()
        mock_groundedness_class = MagicMock()
        mock_openai_class = MagicMock()

        # Setup the module attributes
        mock_modules["trulens_eval"].Tru = mock_tru
        mock_modules["trulens_eval.feedback"].Feedback = mock_feedback
        mock_modules["trulens_eval.feedback"].Groundedness = mock_groundedness_class
        mock_modules["trulens_eval.feedback.provider"].OpenAI = mock_openai_class

        # Mock provider instance
        mock_provider = MagicMock()
        mock_openai_class.return_value = mock_provider

        # Mock groundedness instance
        mock_grounded_instance = MagicMock()
        mock_grounded_instance.groundedness_measure.return_value = 0.8
        mock_groundedness_class.return_value = mock_grounded_instance

        # Mock feedback functions
        mock_ans_feedback = MagicMock()
        mock_ans_feedback.return_value = 0.7
        mock_ctx_feedback = MagicMock()
        mock_ctx_feedback.return_value = 0.9

        mock_feedback.side_effect = [mock_ans_feedback, mock_ctx_feedback]

        with patch.dict("sys.modules", mock_modules):
            with patch("askme.evals.trulens_runner.os") as mock_os:
                mock_os.getenv.return_value = "http://localhost:11434/v1"

                # Provide a dict-like environ that allows tracking setdefault calls
                class MockEnviron(dict):
                    def __init__(self: Any) -> Any:
                        super().__init__()
                        self.setdefault = MagicMock(
                            side_effect=lambda k, v: super(
                                MockEnviron, self
                            ).setdefault(k, v)
                        )

                mock_os.environ = MockEnviron()

                # Test data
                samples = [
                    {
                        "question": "What is machine learning?",
                        "answer": "Machine learning is a type of AI",
                        "contexts": ["ML is a subset of AI", "It uses algorithms"],
                    }
                ]

                result = run_trulens(samples)

                # Verify result structure
                assert isinstance(result, dict)
                assert "groundedness" in result
                assert "answer_relevance" in result
                assert "context_relevance" in result

                # Verify scores are floats
                for score in result.values():
                    assert isinstance(score, float)

    def test_trulens_empty_samples(self: Any) -> None:
        """Test TruLens with empty samples list."""
        mock_modules = {
            "trulens_eval": MagicMock(),
            "trulens_eval.feedback": MagicMock(),
            "trulens_eval.feedback.provider": MagicMock(),
        }

        mock_tru = MagicMock()
        mock_feedback = MagicMock()
        mock_groundedness_class = MagicMock()
        mock_openai_class = MagicMock()

        mock_modules["trulens_eval"].Tru = mock_tru
        mock_modules["trulens_eval.feedback"].Feedback = mock_feedback
        mock_modules["trulens_eval.feedback"].Groundedness = mock_groundedness_class
        mock_modules["trulens_eval.feedback.provider"].OpenAI = mock_openai_class

        with patch.dict("sys.modules", mock_modules):
            with patch("askme.evals.trulens_runner.os") as mock_os:
                mock_os.getenv.return_value = "http://localhost:11434/v1"
                mock_os.environ = {}

                result = run_trulens([])

                # Should return zero scores for empty samples
                expected_keys = [
                    "groundedness",
                    "answer_relevance",
                    "context_relevance",
                ]
                assert all(key in result for key in expected_keys)
                assert all(result[key] == 0.0 for key in expected_keys)

    def test_trulens_evaluation_exceptions(self: Any) -> None:
        """Test TruLens handles individual evaluation exceptions gracefully."""
        mock_modules = {
            "trulens_eval": MagicMock(),
            "trulens_eval.feedback": MagicMock(),
            "trulens_eval.feedback.provider": MagicMock(),
        }

        mock_tru = MagicMock()
        mock_feedback = MagicMock()
        mock_groundedness_class = MagicMock()
        mock_openai_class = MagicMock()

        mock_modules["trulens_eval"].Tru = mock_tru
        mock_modules["trulens_eval.feedback"].Feedback = mock_feedback
        mock_modules["trulens_eval.feedback"].Groundedness = mock_groundedness_class
        mock_modules["trulens_eval.feedback.provider"].OpenAI = mock_openai_class

        # Mock provider
        mock_provider = MagicMock()
        mock_openai_class.return_value = mock_provider

        # Mock groundedness to raise exception
        mock_grounded_instance = MagicMock()
        mock_grounded_instance.groundedness_measure.side_effect = Exception(
            "Groundedness failed"
        )
        mock_groundedness_class.return_value = mock_grounded_instance

        # Mock feedback to raise exceptions
        mock_feedback_func = MagicMock()
        mock_feedback_func.side_effect = Exception("Feedback failed")
        mock_feedback.return_value = mock_feedback_func

        with patch.dict("sys.modules", mock_modules):
            with patch("askme.evals.trulens_runner.os") as mock_os:
                mock_os.getenv.return_value = "http://localhost:11434/v1"
                mock_os.environ = {}

                samples = [
                    {
                        "question": "Test question?",
                        "answer": "Test answer",
                        "contexts": ["Test context"],
                    }
                ]

                result = run_trulens(samples)

                # Should return zero scores when all evaluations fail
                expected_keys = [
                    "groundedness",
                    "answer_relevance",
                    "context_relevance",
                ]
                assert all(key in result for key in expected_keys)
                assert all(result[key] == 0.0 for key in expected_keys)

    def test_trulens_provider_initialization_error(self: Any) -> None:
        """Test RuntimeError when provider initialization fails."""
        mock_modules = {
            "trulens_eval": MagicMock(),
            "trulens_eval.feedback": MagicMock(),
            "trulens_eval.feedback.provider": MagicMock(),
        }

        mock_tru = MagicMock()
        mock_feedback = MagicMock()
        mock_groundedness_class = MagicMock()
        mock_openai_class = MagicMock()

        mock_modules["trulens_eval"].Tru = mock_tru
        mock_modules["trulens_eval.feedback"].Feedback = mock_feedback
        mock_modules["trulens_eval.feedback"].Groundedness = mock_groundedness_class
        mock_modules["trulens_eval.feedback.provider"].OpenAI = mock_openai_class

        # Mock provider initialization to fail
        mock_openai_class.side_effect = Exception("Provider initialization failed")

        with patch.dict("sys.modules", mock_modules):
            with patch("askme.evals.trulens_runner.os") as mock_os:
                mock_os.getenv.return_value = "http://localhost:11434/v1"
                mock_os.environ = {}

                samples = [{"question": "test", "answer": "test", "contexts": ["test"]}]

                with pytest.raises(RuntimeError, match="TruLens evaluation failed"):
                    run_trulens(samples)

    def test_trulens_environment_setup(self: Any) -> None:
        """Test proper environment variable setup."""
        mock_modules = {
            "trulens_eval": MagicMock(),
            "trulens_eval.feedback": MagicMock(),
            "trulens_eval.feedback.provider": MagicMock(),
        }

        mock_tru = MagicMock()
        mock_feedback = MagicMock()
        mock_groundedness_class = MagicMock()
        mock_openai_class = MagicMock()

        mock_modules["trulens_eval"].Tru = mock_tru
        mock_modules["trulens_eval.feedback"].Feedback = mock_feedback
        mock_modules["trulens_eval.feedback"].Groundedness = mock_groundedness_class
        mock_modules["trulens_eval.feedback.provider"].OpenAI = mock_openai_class

        with patch.dict("sys.modules", mock_modules):
            with patch("askme.evals.trulens_runner.os") as mock_os:
                mock_os.getenv.side_effect = lambda key, default=None: {
                    "OPENAI_BASE_URL": "http://test:8080/v1",
                    "OPENAI_API_KEY": "test-key",
                }.get(key, default)

                class MockEnviron(dict):
                    def __init__(self: Any) -> Any:
                        super().__init__()
                        self.setdefault = MagicMock(
                            side_effect=lambda k, v: super(
                                MockEnviron, self
                            ).setdefault(k, v)
                        )

                mock_environ = MockEnviron()
                mock_os.environ = mock_environ

                run_trulens([])

                # Verify environment variables were set
                assert mock_environ.get("OPENAI_API_BASE") == "http://test:8080/v1"
                mock_environ.setdefault.assert_called()

    def test_trulens_multiple_samples_averaging(self: Any) -> None:
        """Test TruLens with multiple samples and score averaging."""
        mock_modules = {
            "trulens_eval": MagicMock(),
            "trulens_eval.feedback": MagicMock(),
            "trulens_eval.feedback.provider": MagicMock(),
        }

        mock_tru = MagicMock()
        mock_feedback = MagicMock()
        mock_groundedness_class = MagicMock()
        mock_openai_class = MagicMock()

        mock_modules["trulens_eval"].Tru = mock_tru
        mock_modules["trulens_eval.feedback"].Feedback = mock_feedback
        mock_modules["trulens_eval.feedback"].Groundedness = mock_groundedness_class
        mock_modules["trulens_eval.feedback.provider"].OpenAI = mock_openai_class

        # Mock provider
        mock_provider = MagicMock()
        mock_openai_class.return_value = mock_provider

        # Mock groundedness with different scores
        mock_grounded_instance = MagicMock()
        mock_grounded_instance.groundedness_measure.side_effect = [0.8, 0.6]
        mock_groundedness_class.return_value = mock_grounded_instance

        # Mock feedback functions with different scores
        mock_ans_feedback = MagicMock()
        mock_ans_feedback.side_effect = [0.7, 0.9]
        mock_ctx_feedback = MagicMock()
        mock_ctx_feedback.side_effect = [0.5, 0.7]

        mock_feedback.side_effect = [mock_ans_feedback, mock_ctx_feedback]

        with patch.dict("sys.modules", mock_modules):
            with patch("askme.evals.trulens_runner.os") as mock_os:
                mock_os.getenv.return_value = "http://localhost:11434/v1"
                mock_os.environ = {}

                # Test with two samples
                samples = [
                    {
                        "question": "Question 1?",
                        "answer": "Answer 1",
                        "contexts": ["Context 1"],
                    },
                    {
                        "question": "Question 2?",
                        "answer": "Answer 2",
                        "contexts": ["Context 2"],
                    },
                ]

                result = run_trulens(samples)

                # Verify averaging works correctly
                assert abs(result["groundedness"] - 0.7) < 0.001  # (0.8 + 0.6) / 2
                assert abs(result["answer_relevance"] - 0.8) < 0.001  # (0.7 + 0.9) / 2
                assert abs(result["context_relevance"] - 0.6) < 0.001  # (0.5 + 0.7) / 2

    def test_trulens_with_provider_parameter(self: Any) -> None:
        """Test TruLens with provider parameter (currently unused)."""
        # Test that provider parameter is accepted but currently doesn't change behavior
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            with pytest.raises(RuntimeError, match="TruLens not available"):
                run_trulens([], provider="openai")

    def test_trulens_missing_sample_fields(self: Any) -> None:
        """Test TruLens with samples missing some fields."""
        mock_modules = {
            "trulens_eval": MagicMock(),
            "trulens_eval.feedback": MagicMock(),
            "trulens_eval.feedback.provider": MagicMock(),
        }

        mock_tru = MagicMock()
        mock_feedback = MagicMock()
        mock_groundedness_class = MagicMock()
        mock_openai_class = MagicMock()

        mock_modules["trulens_eval"].Tru = mock_tru
        mock_modules["trulens_eval.feedback"].Feedback = mock_feedback
        mock_modules["trulens_eval.feedback"].Groundedness = mock_groundedness_class
        mock_modules["trulens_eval.feedback.provider"].OpenAI = mock_openai_class

        # Mock provider
        mock_provider = MagicMock()
        mock_openai_class.return_value = mock_provider

        # Mock instances
        mock_grounded_instance = MagicMock()
        mock_grounded_instance.groundedness_measure.return_value = 0.8
        mock_groundedness_class.return_value = mock_grounded_instance

        mock_feedback_func = MagicMock()
        mock_feedback_func.return_value = 0.7
        mock_feedback.return_value = mock_feedback_func

        with patch.dict("sys.modules", mock_modules):
            with patch("askme.evals.trulens_runner.os") as mock_os:
                mock_os.getenv.return_value = "http://localhost:11434/v1"
                mock_os.environ = {}

                # Sample with missing fields
                samples = [
                    {
                        # Missing question, answer, contexts - should default to empty strings/lists
                    }
                ]

                result = run_trulens(samples)

                # Should handle missing fields gracefully
                assert isinstance(result, dict)
                assert all(
                    key in result
                    for key in ["groundedness", "answer_relevance", "context_relevance"]
                )

    def test_trulens_context_joining(self: Any) -> None:
        """Test that contexts are properly joined for context relevance evaluation."""
        mock_modules = {
            "trulens_eval": MagicMock(),
            "trulens_eval.feedback": MagicMock(),
            "trulens_eval.feedback.provider": MagicMock(),
        }

        mock_tru = MagicMock()
        mock_feedback = MagicMock()
        mock_groundedness_class = MagicMock()
        mock_openai_class = MagicMock()

        mock_modules["trulens_eval"].Tru = mock_tru
        mock_modules["trulens_eval.feedback"].Feedback = mock_feedback
        mock_modules["trulens_eval.feedback"].Groundedness = mock_groundedness_class
        mock_modules["trulens_eval.feedback.provider"].OpenAI = mock_openai_class

        # Mock provider
        mock_provider = MagicMock()
        mock_openai_class.return_value = mock_provider

        # Track what context text is passed to context relevance
        mock_ctx_feedback = MagicMock()
        mock_ctx_feedback.return_value = 0.8
        mock_ans_feedback = MagicMock()
        mock_ans_feedback.return_value = 0.7

        mock_feedback.side_effect = [mock_ans_feedback, mock_ctx_feedback]

        with patch.dict("sys.modules", mock_modules):
            with patch("askme.evals.trulens_runner.os") as mock_os:
                mock_os.getenv.return_value = "http://localhost:11434/v1"
                mock_os.environ = {}

                samples = [
                    {
                        "question": "Test question?",
                        "answer": "Test answer",
                        "contexts": ["Context 1", "Context 2", "Context 3"],
                    }
                ]

                run_trulens(samples)

                # Verify context relevance was called with joined contexts
                mock_ctx_feedback.assert_called_with(
                    "Context 1\nContext 2\nContext 3", "Test question?"
                )
