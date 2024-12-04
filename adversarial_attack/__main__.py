import argparse
import sys

import torch.nn
from PIL import Image
import numpy as np

<<<<<<< Updated upstream

from .resnet_utils import (
=======
from adversarial_attack.resnet_utils import (
>>>>>>> Stashed changes
    AVAILABLE_MODELS,
    load_model_default_weights,
    get_model_categories,
    category_to_tensor,
    load_image,
    preprocess_image,
    to_array,
)
<<<<<<< Updated upstream
from .fgsm import get_attack_fn
=======
from adversarial_attack.api import perform_attack
>>>>>>> Stashed changes


def main():
    parser = argparse.ArgumentParser(
        description="Run adversarial attack on a PyTorch model with a given image."
    )
    parser.add_argument(
        "--model",
        "-m",
        help=f"Name of the PyTorch model to attack. Available models: {AVAILABLE_MODELS.keys()}",
        required=True,
    )
    parser.add_argument(
        "--mode",
        default="targeted",
        help=f"Mode of attack. Options: standard, targeted. Default: targeted.",
        required=False,
    )
    parser.add_argument(
        "--image",
        "-i",
        help="Path to the image to attack.",
        required=False,
    )
    parser.add_argument(
        "--category-truth",
        "-c",
        help="String representing the true category of the image.",
        required=True,
    )
    parser.add_argument(
        "--category-target",
        "-ct",
        help="String representing the true category of the image.",
        required=True if "targeted" in sys.argv else False,  # required only for targeted attacks
    )
    parser.add_argument(
        "--epsilon",
        "-eps",
        default=1.0e-3,
        help="Epsilon value for the FGSM attack.",
        required=False,
    )
    parser.add_argument(
        "--max-iterations",
        "-it",
        default=50,
        help="Maximum number of iterations for the FGSM attack.",
        required=False,
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to save the adversarial image.",
        required=False,
    )

    args = parser.parse_args()

    if args.mode == "targeted" and args.category_target is None:
        raise ValueError("Target category is required for targeted attacks.")

    model = load_model_default_weights(model_name=args.model)
    model.eval()  # IMPORTANT: set model to evaluation mode

    print(model)

    image = load_image(args.image)
    image_tensor = preprocess_image(image)

    categories = get_model_categories(args.model)
    attack_fn = get_attack_fn(
        mode=args.mode,
        model=model,
        tensor=image_tensor,
        category_truth=args.category_truth,
        category_target=args.category_target,
        categories=categories,
        epsilon=args.epsilon,
        max_iter=int(args.max_iterations),
    )
    results = attack_fn()

    if results is not None:
        new_image, orig_pred, new_pred = results
        print("Adversarial attack succeeded!")
        print(f"Original Prediction: {categories[orig_pred.argmax().item()]}")
        print(f"New Prediction: {categories[new_pred.item()]}")

        if args.output is not None:
            Image.fromarray(np.uint8(255 * to_array(new_image))).save(args.output)
    else:
        print("Adversarial attack failed. Won't save image if output path is provided.")


if __name__ == "__main__":
    main()
