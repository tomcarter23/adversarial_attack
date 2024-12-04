
import typing as ty

import torch.nn
from PIL import Image
from .fgsm import get_attack_fn


def perform_attack(
    mode: str,
    model: torch.nn.Module,
    image: torch.Tensor,
    categories: list[str],
    true_category: str,
    target_category: str = None,
    epsilon: float = 1.0e-3,
    max_iter: int = 50,
) -> ty.Optional[torch.Tensor]:
    """
    API to perform an adversarial attack on a PyTorch model with a given image.

    Args:
        mode (str): Mode of attack.
        model (torch.nn.Module): PyTorch model to attack.
        image (Image): Image to attack.
        categories (list[str]): List of categories for the model.
        true_category (str): True category of the image.
        target_category (str): Optional target category for targeted attacks.
        epsilon (float): Epsilon value for the FGSM attack.
        max_iter (int): Maximum number of iterations for the FGSM attack.

    Returns:
        Image: Adversarial image or None if attack failed.
    """
    model.eval()  # IMPORTANT: set model to evaluation mode

    attack_fn = get_attack_fn(
        mode=mode,
        model=model,
        tensor=image,
        category_truth=true_category,
        category_target=target_category,
        categories=categories,
        epsilon=epsilon,
        max_iter=max_iter,
    )
    results = attack_fn()

    if results is not None:
        new_image, orig_pred_idx, new_pred_idx = results
        print("Adversarial attack succeeded!")
        print(f"Original Prediction: {categories[orig_pred_idx]}")
        print(f"New Prediction: {categories[new_pred_idx]}")

        return new_image

    print("Adversarial attack failed.")
    return None
