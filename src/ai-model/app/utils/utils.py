import os
import logging
import yaml
import torch
from torchvision import transforms
from io import BytesIO
from PIL import Image
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse a YAML configs file.

    Args:
        config_path (str): Path to the YAML configs file.

    Returns:
        Dict[str, Any]: Parsed configs as a dictionary.
    """
    logger.info(f"Loading configs from {config_path}")
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logger.info("Configuration loaded successfully.")
        return config
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        raise


def extract_config_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts and validates configs values from the loaded dictionary.

    Args:
        config (Dict[str, Any]): The parsed YAML configs dictionary.

    Returns:
        Dict[str, Any]: Extracted values including paths, hyperparameters, and training settings.
    """
    try:
        # Path configurations
        data_dir: str = config["paths"]["data_dir"]
        model_path: str = config["paths"]["model_path"]

        # Hyperparameters
        batch_size: int = config["hyperparameters"]["batch_size"]
        num_epochs: int = config["hyperparameters"]["num_epochs"]
        learning_rate: float = config["hyperparameters"]["learning_rate"]
        num_classes: int = config["hyperparameters"]["num_classes"]

        # Training settings
        image_size: Tuple[int, int] = tuple(config["training"]["image_size"])
        use_pretrained: bool = config["training"]["use_pretrained"]

        logger.info("Configuration values extracted and validated.")
        return {
            "data_dir": data_dir,
            "model_path": model_path,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "num_classes": num_classes,
            "image_size": image_size,
            "use_pretrained": use_pretrained,
        }

    except KeyError as e:
        logger.error(f"Missing configs key: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to extract config values: {e}")
        raise


def get_default_transform(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Creates a standard image transformation pipeline for preprocessing images.

    Args:
        image_size (Tuple[int, int], optional): The target size to which input images will be resized, given as (height,
            width). Defaults to (224, 224).

    Returns:
        transforms.Compose: A composed transform that applies resizing, tensor conversion, and normalization in
            sequence.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def read_image(image_bytes: bytes) -> torch.Tensor:
    """
    Converts raw image bytes into a preprocessed PyTorch tensor suitable for model prediction.

    Parameters:
        image_bytes (bytes): The raw image bytes.

    Returns:
        torch.Tensor: The transformed image tensor, normalized and resized for model input.

    Raises:
        Exception: If the image cannot be opened or transformed.
    """
    logger.info("Reading image from bytes")
    try:
        # Open image from bytes and ensure it's in RGB format
        image: Image.Image = Image.open(BytesIO(image_bytes)).convert("RGB")
        logger.info(f"Original image size: {image.size}")

        # Apply preprocessing pipeline
        transform = get_default_transform()
        tensor: torch.Tensor = transform(image)
        logger.info(f"Image transformed to tensor with shape: {tensor.shape}")

        return tensor

    except Exception as e:
        logger.error(f"Failed to read or transform image: {e}")
        raise
