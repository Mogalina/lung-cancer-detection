import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

logger = logging.getLogger(__name__)


class LungCancerModel:
    """
    Represents a lung cancer classification model based on ResNet50.

    Attributes:
        device (torch.device): Device to run the model on (CPU or GPU).
        num_classes (int): Number of output classes for classification.
        model_path (str | None): File path to save/load the model weights.
        model (torch.nn.Module): The PyTorch model instance.
    """

    def __init__(
            self,
            num_classes: int,
            device: torch.device | None = None,
            model_path: str | None = None,
            use_pretrained: bool = False
    ) -> None:
        """
        Initialize the model with specified parameters.

        Args:
            num_classes (int): Number of output classes.
            device (torch.device | None, optional): Device to run the model on. If None, will auto-select GPU if
                available, else CPU. Defaults to None.
            model_path (str | None, optional): Path to save or load the model weights. Defaults to None.
            use_pretrained (bool, optional): Whether to initialize ResNet50 with pretrained weights. Defaults to False.
        """
        # Choose device
        self.device: torch.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes: int = num_classes
        self.model_path: str | None = model_path

        # Choose weights
        if use_pretrained:
            weights = ResNet50_Weights.DEFAULT
            logger.info("Using pretrained ResNet50 weights from torchvision.")
        else:
            weights = None
            logger.info("Using randomly initialized ResNet50 weights.")

        # Build model
        self.model: torch.nn.Module = models.resnet50(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(self.device)

        # Load custom weights if available
        if self.model_path and os.path.isfile(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Loaded model weights from {self.model_path}")

    def save(self) -> None:
        """
        Save the model weights to the specified path.

        Raises:
            ValueError: If path is not set.
        """
        if self.model_path is None:
            raise ValueError("Model path not set, cannot save model.")

        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Save the model state dictionary
        torch.save(self.model.state_dict(), self.model_path)
        logger.info(f"Model saved to {self.model_path}")

    def load(self) -> None:
        """
        Load model weights from the specified path.

        Raises:
            FileNotFoundError: If the model file does not exist at path.
        """
        if self.model_path is None or not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded from {self.model_path}")

    def predict(self, image_tensor: torch.Tensor) -> dict[str, str | float]:
        """
        Predict the class label and confidence score for a single image tensor.

        Args:
            image_tensor (torch.Tensor): Image tensor with shape [C, H, W].

        Returns:
            dict[str, str | float]: Dictionary containing:
                - "predicted_class" (str): Predicted class label.
                - "confidence" (float): Confidence score of the prediction.
        """
        self.model.eval()
        with torch.no_grad():
            # Add batch dimension and move tensor to the device
            input_tensor: torch.Tensor = image_tensor.unsqueeze(0).to(self.device)

            # Forward pass through the model
            outputs: torch.Tensor = self.model(input_tensor)

            # Apply softmax to get class probabilities
            probs: torch.Tensor = F.softmax(outputs, dim=1).cpu().numpy()[0]

            # Get predicted class index and confidence
            pred_class_idx: int = probs.argmax()
            confidence: float = float(probs[pred_class_idx])

            # Map class index to human-readable labels
            class_labels: dict[int, str] = {0: "benign", 1: "malignant", 2: "normal"}
            pred_class: str = class_labels.get(pred_class_idx, "unknown")

            logger.info(f"Prediction: {pred_class} with confidence {confidence:.4f}")

        return {"label": pred_class, "confidence": confidence}
