from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import torchvision.models as models


AVAILABLE_MODELS = {
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "resnet101": "ResNet101_Weights",
    "resnet152": "ResNet152_Weights",
}


def load_model_default_weights(model_name: str):
    """
    Load a model from a given GitHub repository.

    Args:
        model_name (str): Name of the model to load.

    Returns:
        torch.Model: Loaded PyTorch model.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found.")

    weights = getattr(models, AVAILABLE_MODELS[model_name]).DEFAULT
    return getattr(models, model_name)(weights=weights)


def get_model_categories(model_name: str) -> list[str]:
    """
    Get the categories of models available.

    Returns:
        list[str]: List of model categories.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found.")

    return getattr(models, AVAILABLE_MODELS[model_name]).DEFAULT.meta["categories"]


def load_image(image_path: str) -> Image:
    """
    Load an image from a file.

    Args:
        image_path (str): Path to the image file.

    Returns:
        Image: Loaded image.
    """
    return Image.open(image_path)


def preprocess_image(image: Image):
    """
    Preprocess an image to be compatible with ResNet models.

    Args:
        image (Image): Image to preprocess.

    Returns:
        Tensor: Preprocessed image.
    """
    # Preprocess the image to be compatible with ResNet models
    # https://pytorch.org/hub/pytorch_vision_resnet/
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(
        0
    )  # create a mini-batch as expected by the model

    return input_batch


def to_array(tensor) -> np.ndarray:
    """
    Convert output tensor to numpy array.

    Args:
        tensor (torch.Tensor): Output tensor.

    Returns:
        np.array: Numpy array.
    """

    # reverse the preprocessing steps
    tensor = tensor.squeeze()

    unnormalize = transforms.Compose(
        [
            transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        ]
    )

    array = unnormalize(tensor).permute(1, 2, 0).detach().numpy()

    return array


def category_to_tensor(categ: str, categs: list[str]) -> torch.Tensor:
    """
    Get category tensor from category string and list of categories.

    Args:
        categ (str): Category string.
        categs (list[str]): List of categories.

    Returns:
        torch.Tensor: Category tensor
    """
    if categ not in categs:
        raise ValueError(f"Category {categ} not found in categories.")
    return torch.tensor([categs.index(categ)])
