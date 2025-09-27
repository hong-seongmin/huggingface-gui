"""
Basic code snippet classes and enums.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any


class SnippetType(Enum):
    """Code snippet type classification."""
    LOADING = "loading"
    INFERENCE = "inference"
    BATCH_INFERENCE = "batch_inference"
    FINE_TUNING = "fine_tuning"
    OPTIMIZATION = "optimization"
    DEPLOYMENT = "deployment"


@dataclass
class CodeSnippet:
    """Code snippet information container."""
    title: str
    description: str
    code: str
    language: str
    snippet_type: SnippetType
    requirements: List[str]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'title': self.title,
            'description': self.description,
            'code': self.code,
            'language': self.language,
            'snippet_type': self.snippet_type.value,
            'requirements': self.requirements,
            'notes': self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeSnippet':
        """Create from dictionary."""
        return cls(
            title=data['title'],
            description=data['description'],
            code=data['code'],
            language=data['language'],
            snippet_type=SnippetType(data['snippet_type']),
            requirements=data.get('requirements', []),
            notes=data.get('notes', [])
        )

    def format_for_display(self) -> str:
        """Format snippet for display."""
        return f"""
# {self.title}

{self.description}

```{self.language}
{self.code}
```

Requirements: {', '.join(self.requirements) if self.requirements else 'None'}

Notes:
{chr(10).join('- ' + note for note in self.notes) if self.notes else '- None'}
"""

    def get_formatted_requirements(self) -> str:
        """Get formatted requirements string."""
        if not self.requirements:
            return "# No additional requirements"

        return "# Requirements:\n" + "\n".join(f"# pip install {req}" for req in self.requirements)

    def get_formatted_notes(self) -> str:
        """Get formatted notes string."""
        if not self.notes:
            return "# No additional notes"

        return "# Notes:\n" + "\n".join(f"# - {note}" for note in self.notes)


class SnippetTemplate:
    """Base class for code snippet templates."""

    def __init__(self, model_id: str, task_type: str = None, model_class: str = None):
        """
        Initialize template.

        Args:
            model_id: Model identifier (HuggingFace ID or local path)
            task_type: Task type for the model
            model_class: Model class name
        """
        self.model_id = model_id
        self.task_type = task_type or "unknown"
        self.model_class = model_class or "AutoModel"

    def generate(self) -> CodeSnippet:
        """Generate code snippet (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement generate method")

    def _format_model_id(self) -> str:
        """Format model ID for code generation."""
        if self._is_local_path(self.model_id):
            return f'"{self.model_id}"'
        else:
            return f'"{self.model_id}"'

    def _is_local_path(self, path: str) -> bool:
        """Check if path is a local file path."""
        return path.startswith('/') or path.startswith('./') or '\\' in path

    def _get_base_imports(self) -> List[str]:
        """Get base import statements."""
        return [
            "from transformers import AutoTokenizer, AutoModel",
            "import torch"
        ]

    def _get_task_specific_imports(self) -> List[str]:
        """Get task-specific import statements."""
        task_imports = {
            'text-classification': ["from transformers import AutoModelForSequenceClassification"],
            'token-classification': ["from transformers import AutoModelForTokenClassification"],
            'question-answering': ["from transformers import AutoModelForQuestionAnswering"],
            'text-generation': ["from transformers import AutoModelForCausalLM"],
            'text2text-generation': ["from transformers import AutoModelForSeq2SeqLM"],
            'summarization': ["from transformers import AutoModelForSeq2SeqLM"],
            'translation': ["from transformers import AutoModelForSeq2SeqLM"],
            'feature-extraction': ["from transformers import AutoModel"],
            'fill-mask': ["from transformers import AutoModelForMaskedLM"],
            'image-classification': ["from transformers import AutoImageProcessor", "from PIL import Image"],
            'automatic-speech-recognition': ["import librosa", "import soundfile as sf"],
        }

        return task_imports.get(self.task_type, [])


class SnippetValidator:
    """Validates code snippets for correctness and completeness."""

    @staticmethod
    def validate_snippet(snippet: CodeSnippet) -> List[str]:
        """
        Validate a code snippet.

        Args:
            snippet: CodeSnippet to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required fields
        if not snippet.title:
            errors.append("Title is required")

        if not snippet.description:
            errors.append("Description is required")

        if not snippet.code:
            errors.append("Code is required")

        if not snippet.language:
            errors.append("Language is required")

        # Check code syntax (basic check)
        if snippet.language == "python":
            try:
                compile(snippet.code, '<string>', 'exec')
            except SyntaxError as e:
                errors.append(f"Python syntax error: {e}")

        # Check for common issues
        if snippet.language == "python" and "import" in snippet.code:
            if not any(req in snippet.code for req in ["transformers", "torch"]):
                errors.append("Code appears to be missing required imports")

        return errors

    @staticmethod
    def validate_snippets(snippets: Dict[str, CodeSnippet]) -> Dict[str, List[str]]:
        """
        Validate multiple snippets.

        Args:
            snippets: Dictionary of snippets to validate

        Returns:
            Dictionary mapping snippet names to validation errors
        """
        validation_results = {}

        for name, snippet in snippets.items():
            errors = SnippetValidator.validate_snippet(snippet)
            if errors:
                validation_results[name] = errors

        return validation_results


class SnippetMetadata:
    """Metadata for code snippets."""

    def __init__(self, model_id: str, task_type: str, framework: str = "transformers"):
        """
        Initialize metadata.

        Args:
            model_id: Model identifier
            task_type: Task type
            framework: Framework being used
        """
        self.model_id = model_id
        self.task_type = task_type
        self.framework = framework
        self.tags = []
        self.difficulty = "beginner"
        self.estimated_runtime = "unknown"
        self.memory_requirements = "unknown"

    def add_tag(self, tag: str):
        """Add a tag to metadata."""
        if tag not in self.tags:
            self.tags.append(tag)

    def set_difficulty(self, difficulty: str):
        """Set difficulty level (beginner, intermediate, advanced)."""
        self.difficulty = difficulty

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_id': self.model_id,
            'task_type': self.task_type,
            'framework': self.framework,
            'tags': self.tags,
            'difficulty': self.difficulty,
            'estimated_runtime': self.estimated_runtime,
            'memory_requirements': self.memory_requirements
        }