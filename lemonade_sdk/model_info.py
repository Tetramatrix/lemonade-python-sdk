"""
Module for model metadata and capability detection

Provides the ModelInfo class which wraps Lemonade model responses
with convenient methods to check capabilities via labels.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


# Official Lemonade label constants
LABEL_VISION = "vision"
LABEL_REASONING = "reasoning"
LABEL_CODING = "coding"
LABEL_TOOL_CALLING = "tool-calling"
LABEL_EMBEDDINGS = "embeddings"
LABEL_RERANKING = "reranking"
LABEL_IMAGE = "image"
LABEL_HOT = "hot"
LABEL_CUSTOM = "custom"

# Human-readable label descriptions
LABEL_DESCRIPTIONS: Dict[str, str] = {
    LABEL_VISION: "Model supports image input (VLM)",
    LABEL_REASONING: "Model uses extended thinking/chain-of-thought",
    LABEL_CODING: "Optimized for code generation tasks",
    LABEL_TOOL_CALLING: "Supports function/tool calling",
    LABEL_EMBEDDINGS: "Text embedding model",
    LABEL_RERANKING: "Reranking model (for RAG pipelines)",
    LABEL_IMAGE: "Image generation model (Stable Diffusion etc.)",
    LABEL_HOT: "Featured/recommended by Lemonade",
    LABEL_CUSTOM: "User-added model",
}


@dataclass
class ModelInfo:
    """
    Represents metadata and capabilities for a Lemonade model.

    Wraps the raw API response from /api/v1/models and provides
    convenient methods to check model capabilities via labels.

    Attributes:
        id: The model identifier (e.g., "user.Qwen3.5-122B-A10B-UD-IQ3_S.gguf")
        name: Human-readable model name
        labels: List of capability labels (e.g., ["custom", "vision"])
        recipe: The backend recipe (e.g., "llamacpp", "mistralrs")
        raw: The full raw API response dict
        mmproj: Path to the mmproj file for vision models (optional)
    """

    id: str
    name: str
    labels: List[str] = field(default_factory=list)
    recipe: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
    mmproj: Optional[str] = None

    # ---- Label helpers ----

    def has_label(self, label: str) -> bool:
        """Check if the model has a specific label."""
        return label in self.labels

    def has_vision(self) -> bool:
        """Check if the model supports image input (VLM)."""
        return LABEL_VISION in self.labels

    def has_reasoning(self) -> bool:
        """Check if the model uses extended thinking/chain-of-thought."""
        return LABEL_REASONING in self.labels

    def has_coding(self) -> bool:
        """Check if the model is optimized for code generation."""
        return LABEL_CODING in self.labels

    def has_tool_calling(self) -> bool:
        """Check if the model supports function/tool calling."""
        return LABEL_TOOL_CALLING in self.labels

    def has_embeddings(self) -> bool:
        """Check if the model is a text embedding model."""
        return LABEL_EMBEDDINGS in self.labels

    def has_reranking(self) -> bool:
        """Check if the model is a reranking model."""
        return LABEL_RERANKING in self.labels

    def has_image_generation(self) -> bool:
        """Check if the model supports image generation (Stable Diffusion etc.)."""
        return LABEL_IMAGE in self.labels

    def is_hot(self) -> bool:
        """Check if the model is featured/recommended by Lemonade."""
        return LABEL_HOT in self.labels

    def is_custom(self) -> bool:
        """Check if the model is user-added."""
        return LABEL_CUSTOM in self.labels

    def get_label_descriptions(self) -> List[str]:
        """Get human-readable descriptions for all model labels."""
        return [
            LABEL_DESCRIPTIONS[label]
            for label in self.labels
            if label in LABEL_DESCRIPTIONS
        ]

    def get_capabilities_summary(self) -> str:
        """Get a human-readable summary of the model's capabilities."""
        if not self.labels:
            return "No capabilities listed"

        descriptions = self.get_label_descriptions()
        return "; ".join(descriptions)

    # ---- Factory ----

    @classmethod
    def from_api_response(cls, model_data: Dict[str, Any]) -> "ModelInfo":
        """
        Create a ModelInfo from a raw API response dict.

        Args:
            model_data: The model dict from /api/v1/models response

        Returns:
            ModelInfo instance
        """
        model_id = model_data.get("id", model_data.get("name", "unknown"))
        model_name = model_data.get("name", model_data.get("id", "unknown"))
        labels = model_data.get("labels", [])
        recipe = model_data.get("recipe", "")
        mmproj = model_data.get("mmproj")

        return cls(
            id=model_id,
            name=model_name,
            labels=labels,
            recipe=recipe,
            raw=model_data,
            mmproj=mmproj,
        )

    def __repr__(self) -> str:
        labels_str = ", ".join(self.labels) if self.labels else "no labels"
        return f"ModelInfo(id={self.id!r}, labels=[{labels_str}])"
