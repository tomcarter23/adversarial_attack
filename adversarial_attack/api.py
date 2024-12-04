
import typing as ty

import torch.nn
from PIL import Image
import numpy as np

from .resnet_utils import (
    preprocess_image,
    to_array,
)
from .fgsm import get_attack_fn


def perform_attack(
    mode: str,
    model: torch.nn.Module,
    image: Image,
    categories: list[str],
    true_category: str,
    epsilon: float = 1.0e-3,
    max_iter: int = 50,
    target_category: str = None,
) -> ty.Optional[Image]:
    """
    API to perform an adversarial attack on a PyTorch model with a given image.

    Args:
        mode (str): Mode of attack.
        model (torch.nn.Module): PyTorch model to attack.
        image (Image): Image to attack.
        categories (list[str]): List of categories for the model.
        true_category (str): True category of the image.
        epsilon (float): Epsilon value for the FGSM attack.
        max_iter (int): Maximum number of iterations for the FGSM attack.
        target_category (str): Optional target category for targeted attacks.

    Returns:
        Image: Adversarial image or None if attack failed.
    """

    image_tensor = preprocess_image(image)

    attack_fn = get_attack_fn(
        mode=mode,
        model=model,
        tensor=image_tensor,
        category_truth=true_category,
        category_target=target_category,
        categories=categories,
        epsilon=epsilon,
        max_iter=max_iter,
    )
    results = attack_fn()

    if results is not None:
        new_image, orig_pred, new_pred = results
        print("Adversarial attack succeeded!")
        print(f"Original Prediction: {categories[orig_pred.argmax().item()]}")
        print(f"New Prediction: {categories[new_pred.item()]}")

        return Image.fromarray(np.uint8(255 * to_array(new_image)))

    print("Adversarial attack failed.")
    return None
