import torch
import contextlib
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from app.model.model import LungCancerModel
from typing import Tuple, Dict
from torchinfo import summary
from app.utils.utils import *
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize

logger: logging.Logger = logging.getLogger(__name__)


class LungCancerModelService:
    """
    Service class responsible for training, validating, loading, and using a lung cancer classification model.

    This class handles the entire machine learning workflow:
    - Loading configs and datasets
    - Initializing the model, criterion, optimizer, and transforms
    - Training and validating the model
    - Saving, loading, and using the model for predictions
    """

    def __init__(self, config_path: str):
        """
        Initialize the service with configurations.

        Args:
            config_path (str): Path to the configs file.
        """
        # Load and extract configuration using utility functions
        self.config: dict = load_config(config_path)
        config_values: dict = extract_config_values(self.config)

        # Set device (GPU if available, otherwise CPU)
        if torch.cuda.is_available():
            self.device: torch.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device: torch.device = torch.device("mps")
        else:
            self.device: torch.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")

        # Assign config values
        self.data_dir: str = config_values["data_dir"]
        self.train_dir: str = os.path.join(self.data_dir, "split", "train")
        self.val_dir: str = os.path.join(self.data_dir, "split", "val")
        self.model_path: str = config_values["model_path"]
        self.batch_size: int = config_values["batch_size"]
        self.num_epochs: int = config_values["num_epochs"]
        self.learning_rate: float = config_values["learning_rate"]
        self.num_classes: int = config_values["num_classes"]
        self.image_size: Tuple[int, int] = config_values["image_size"]
        self.use_pretrained: bool = config_values["use_pretrained"]

        # Define image transformation pipeline
        self.transform: transforms.Compose = get_default_transform(image_size=self.image_size)

        # Load datasets and data loaders
        self.train_dataset: datasets.ImageFolder = datasets.ImageFolder(self.train_dir, transform=self.transform)
        self.val_dataset: datasets.ImageFolder = datasets.ImageFolder(self.val_dir, transform=self.transform)

        self.train_loader: DataLoader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader: DataLoader = DataLoader(self.val_dataset, batch_size=self.batch_size)

        # Initialize model, loss function, and optimizer
        self.model: LungCancerModel = LungCancerModel(
            num_classes=self.num_classes,
            device=self.device,
            model_path=self.model_path,
            use_pretrained=self.use_pretrained,
        )
        self.criterion: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
        self.optimizer: Adam = Adam(self.model.model.parameters(), lr=self.learning_rate)

        logger.info(f"Training samples: {len(self.train_dataset)}, Validation samples: {len(self.val_dataset)}")

    def train_one_epoch(self) -> float:
        """
        Perform one epoch of training.

        Returns:
            float: The average loss over the training dataset.
        """
        self.model.model.train()
        running_loss: float = 0.0
        correct: int = 0
        total: int = 0

        for images, labels in tqdm(self.train_loader, desc="Training"):
            # Move data to device
            images: torch.Tensor = images.to(self.device)
            labels: torch.Tensor = labels.to(self.device)

            # Forward and backward pass
            self.optimizer.zero_grad()
            outputs: torch.Tensor = self.model.model(images)
            loss: torch.Tensor = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # Accumulate loss and accuracy
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss: float = running_loss / len(self.train_loader)
        accuracy: float = 100 * correct / total
        logger.info(f"Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_loss

    def validate(self) -> float:
        """
        Evaluate the model on the validation set.

        Returns:
            float: The accuracy on the validation dataset.
        """
        self.model.model.eval()
        correct: int = 0
        total: int = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images: torch.Tensor = images.to(self.device)
                labels: torch.Tensor = labels.to(self.device)
                outputs: torch.Tensor = self.model.model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_accuracy: float = 100 * correct / total
        logger.info(f"Validation Accuracy: {val_accuracy:.2f}%")
        return val_accuracy

    def fit(self) -> None:
        """
        Train the model over the configured number of epochs and save it.
        """
        logger.info("Starting training...")
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            self.train_one_epoch()
            self.validate()
        self.model.save()
        self.summarize_model()

    def load_model(self) -> None:
        """
        Load a previously saved model from disk.
        """
        self.model.load()

    def predict(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Predict the class of a single image tensor.

        Args:
            image_tensor (torch.Tensor): The preprocessed image tensor.

        Returns:
            Dict[str, Any]: A dictionary containing prediction results.
        """
        return self.model.predict(image_tensor)

    def summarize_model(
            self,
            input_size: Tuple[int, int, int] = (3, 224, 224),
            output_file: str = "model_summary.txt"
    ) -> None:
        """
        Save a detailed summary of the model architecture.

        Args:
            input_size (Tuple[int, int, int]): Shape of the input tensor (C, H, W).
            output_file (str): Path to the file where the summary will be saved.
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            with contextlib.redirect_stdout(f):
                summary(
                    self.model.model,
                    input_size=(1, *input_size),
                    col_names=["input_size", "output_size", "num_params", "trainable"],
                    device=str(self.device)
                )
        logger.info(f"Model summary saved to {output_file}")

    def test(self) -> float:
        """
        Evaluate the model on the test dataset and compute full evaluation metrics.

        Returns:
            float: The accuracy on the test dataset.
        """
        # Prepare test dataset and loader
        test_dir = os.path.join(self.data_dir, "split", "test")
        test_dataset = datasets.ImageFolder(test_dir, transform=self.transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        logger.info(f"Testing samples: {len(test_dataset)}")
        self.model.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model.model(images)

                probs = torch.softmax(outputs, dim=1)

                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        n_classes = len(test_dataset.classes)

        all_labels_bin = label_binarize(all_labels, classes=range(n_classes))

        acc = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, average='macro')

        try:
            auc = roc_auc_score(all_labels_bin, all_probs, average='macro', multi_class='ovr')
        except ValueError:
            auc = float('nan')

        specificities = []
        for i in range(n_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(specificity)

        avg_specificity = np.mean(specificities)

        logger.info(f"\nConfusion Matrix:\n{cm}")
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"AUC (macro): {auc:.4f}")
        logger.info(f"Recall (Sensitivity, macro): {recall:.4f}")
        logger.info(f"Specificity (average across classes): {avg_specificity:.4f}")

        return acc
