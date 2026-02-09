import sys
import os
from unittest.mock import patch, MagicMock, call

import pytest

# Add src to path so imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import main


@pytest.fixture
def mock_dependencies():
    """Patch all external dependencies used by main()."""
    with (
        patch('main.torch') as mock_torch,
        patch('main.selectTask') as mock_select_task,
        patch('main.divider') as mock_divider,
        patch('main.textGenerationTask') as mock_text_gen,
        patch('main.BitsAndBytesConfig') as mock_bnb,
        patch('main.Confirm') as mock_confirm,
        patch('main.Text') as mock_text,
    ):
        yield {
            'torch': mock_torch,
            'selectTask': mock_select_task,
            'divider': mock_divider,
            'textGenerationTask': mock_text_gen,
            'BitsAndBytesConfig': mock_bnb,
            'Confirm': mock_confirm,
            'Text': mock_text,
        }


class TestMainFirstRun:
    """Tests for main() when user_selected_task is None (first run / new task)."""

    def test_clears_cuda_cache(self, mock_dependencies):
        mocks = mock_dependencies
        mocks['selectTask'].return_value = 'text-generation'
        mocks['textGenerationTask'].return_value = [{'generated_text': []}]
        mocks['Confirm'].ask.return_value = True

        mock_pipeline = MagicMock()

        main(mock_pipeline, 'models/test-model', None, [])

        mocks['torch'].cuda.empty_cache.assert_called_once()

    def test_shows_divider_when_no_prior_task(self, mock_dependencies):
        mocks = mock_dependencies
        mocks['selectTask'].return_value = 'text-generation'
        mocks['textGenerationTask'].return_value = [{'generated_text': []}]
        mocks['Confirm'].ask.return_value = False

        mock_pipeline = MagicMock()

        main(mock_pipeline, 'models/test-model', None, [])

        mocks['divider'].assert_any_call('Select a task for the model to perform')

    def test_calls_select_task_when_no_prior_task(self, mock_dependencies):
        mocks = mock_dependencies
        mocks['selectTask'].return_value = 'text-generation'
        mocks['textGenerationTask'].return_value = [{'generated_text': []}]
        mocks['Confirm'].ask.return_value = True

        mock_pipeline = MagicMock()

        main(mock_pipeline, 'models/test-model', None, [])

        mocks['selectTask'].assert_called_once()

    def test_creates_new_pipeline_with_quantization(self, mock_dependencies):
        mocks = mock_dependencies
        mocks['selectTask'].return_value = 'text-generation'
        mocks['textGenerationTask'].return_value = [{'generated_text': []}]
        mocks['Confirm'].ask.return_value = True

        mock_pipeline = MagicMock()
        model_path = 'models/test-model'

        main(mock_pipeline, model_path, None, [])

        mocks['BitsAndBytesConfig'].assert_called_once_with(load_in_8bit=True)
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args
        assert call_kwargs.kwargs['task'] == 'text-generation'
        assert call_kwargs.kwargs['model'] == model_path
        assert call_kwargs.kwargs['device'] == 0

    def test_calls_text_generation_task(self, mock_dependencies):
        mocks = mock_dependencies
        mocks['selectTask'].return_value = 'text-generation'
        expected_result = [{'generated_text': [{'role': 'assistant', 'content': 'hello'}]}]
        mocks['textGenerationTask'].return_value = expected_result
        mocks['Confirm'].ask.return_value = True

        mock_pipeline = MagicMock()
        created_pipeline = MagicMock()
        mock_pipeline.return_value = created_pipeline

        result = main(mock_pipeline, 'models/test-model', None, [])

        mocks['textGenerationTask'].assert_called_once_with(
            created_pipeline, 'models/test-model', None, 'text-generation', []
        )

    def test_returns_continue_true_when_user_confirms(self, mock_dependencies):
        mocks = mock_dependencies
        mocks['selectTask'].return_value = 'text-generation'
        mocks['textGenerationTask'].return_value = [{'generated_text': []}]
        mocks['Confirm'].ask.return_value = True

        mock_pipeline = MagicMock()

        should_continue, task, gen_text, pipe = main(mock_pipeline, 'models/test-model', None, [])

        assert should_continue is True
        assert task == 'text-generation'

    def test_returns_continue_false_when_user_declines(self, mock_dependencies):
        mocks = mock_dependencies
        mocks['selectTask'].return_value = 'text-generation'
        mocks['textGenerationTask'].return_value = [{'generated_text': []}]
        mocks['Confirm'].ask.return_value = False

        mock_pipeline = MagicMock()

        should_continue, task, gen_text, pipe = main(mock_pipeline, 'models/test-model', None, [])

        assert should_continue is False

    def test_returns_generated_text(self, mock_dependencies):
        mocks = mock_dependencies
        mocks['selectTask'].return_value = 'text-generation'
        expected = [{'generated_text': [{'role': 'assistant', 'content': 'response'}]}]
        mocks['textGenerationTask'].return_value = expected
        mocks['Confirm'].ask.return_value = True

        mock_pipeline = MagicMock()

        _, _, gen_text, _ = main(mock_pipeline, 'models/test-model', None, [])

        assert gen_text == expected

    def test_returns_new_pipeline(self, mock_dependencies):
        mocks = mock_dependencies
        mocks['selectTask'].return_value = 'text-generation'
        mocks['textGenerationTask'].return_value = [{'generated_text': []}]
        mocks['Confirm'].ask.return_value = True

        mock_pipeline = MagicMock()
        created_pipeline = MagicMock()
        mock_pipeline.return_value = created_pipeline

        _, _, _, pipe = main(mock_pipeline, 'models/test-model', None, [])

        assert pipe is created_pipeline


class TestMainContinuation:
    """Tests for main() when user_selected_task is already set (continuation)."""

    def test_does_not_show_divider(self, mock_dependencies):
        mocks = mock_dependencies
        mocks['textGenerationTask'].return_value = [{'generated_text': []}]
        mocks['Confirm'].ask.return_value = True

        existing_pipeline = MagicMock()

        main(existing_pipeline, 'models/test-model', 'text-generation', [])

        mocks['divider'].assert_not_called()

    def test_does_not_call_select_task(self, mock_dependencies):
        mocks = mock_dependencies
        mocks['textGenerationTask'].return_value = [{'generated_text': []}]
        mocks['Confirm'].ask.return_value = True

        existing_pipeline = MagicMock()

        main(existing_pipeline, 'models/test-model', 'text-generation', [])

        mocks['selectTask'].assert_not_called()

    def test_reuses_existing_pipeline(self, mock_dependencies):
        mocks = mock_dependencies
        mocks['textGenerationTask'].return_value = [{'generated_text': []}]
        mocks['Confirm'].ask.return_value = True

        existing_pipeline = MagicMock()

        _, _, _, pipe = main(existing_pipeline, 'models/test-model', 'text-generation', [])

        mocks['BitsAndBytesConfig'].assert_not_called()
        assert pipe is existing_pipeline

    def test_passes_existing_pipeline_to_text_generation(self, mock_dependencies):
        mocks = mock_dependencies
        mocks['textGenerationTask'].return_value = [{'generated_text': []}]
        mocks['Confirm'].ask.return_value = True

        existing_pipeline = MagicMock()
        log = [{'generated_text': [{'role': 'user', 'content': 'hi'}]}]

        main(existing_pipeline, 'models/test-model', 'text-generation', log)

        mocks['textGenerationTask'].assert_called_once_with(
            existing_pipeline, 'models/test-model', 'text-generation', 'text-generation', log
        )


class TestMainUnimplementedTask:
    """Tests for main() when user selects a task that isn't implemented."""

    def test_prints_not_implemented_message(self, mock_dependencies, capsys):
        mocks = mock_dependencies
        mocks['selectTask'].return_value = 'sentiment-analysis'
        mocks['Confirm'].ask.return_value = False

        mock_pipeline = MagicMock()

        main(mock_pipeline, 'models/test-model', None, [])

        captured = capsys.readouterr()
        assert "Task 'sentiment-analysis' is not yet implemented." in captured.out

    def test_unimplemented_task_returns_none_for_generated_text(self, mock_dependencies):
        """The default branch leaves generatedText as None."""
        mocks = mock_dependencies
        mocks['selectTask'].return_value = 'sentiment-analysis'
        mocks['Confirm'].ask.return_value = True

        mock_pipeline = MagicMock()

        _, _, gen_text, _ = main(mock_pipeline, 'models/test-model', None, [])

        assert gen_text is None


class TestMainEdgeCases:
    """Edge case tests."""

    def test_empty_string_task_treated_as_no_task(self, mock_dependencies):
        """An empty string is falsy, so it should trigger task selection."""
        mocks = mock_dependencies
        mocks['selectTask'].return_value = 'text-generation'
        mocks['textGenerationTask'].return_value = [{'generated_text': []}]
        mocks['Confirm'].ask.return_value = False

        mock_pipeline = MagicMock()

        main(mock_pipeline, 'models/test-model', '', [])

        mocks['selectTask'].assert_called_once()

    def test_cuda_cache_cleared_regardless_of_task(self, mock_dependencies):
        mocks = mock_dependencies
        mocks['textGenerationTask'].return_value = [{'generated_text': []}]
        mocks['Confirm'].ask.return_value = True

        existing_pipeline = MagicMock()

        main(existing_pipeline, 'models/test-model', 'text-generation', [])

        mocks['torch'].cuda.empty_cache.assert_called_once()

    def test_divider_called_twice_on_new_pipeline(self, mock_dependencies):
        mocks = mock_dependencies
        mocks['selectTask'].return_value = 'text-generation'
        mocks['textGenerationTask'].return_value = [{'generated_text': []}]
        mocks['Confirm'].ask.return_value = True

        mock_pipeline = MagicMock()

        main(mock_pipeline, 'models/test-model', None, [])

        # First divider for "Select a task", second for "Current Task"
        divider_calls = mocks['divider'].call_args_list
        assert len(divider_calls) == 2
        assert divider_calls[0].args[0] == 'Select a task for the model to perform'
