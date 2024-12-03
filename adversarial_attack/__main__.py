import argparse
from PIL import Image
import numpy as np

from .resnet_utils import (
    AVAILABLE_MODELS,
    load_model_default_weights,
    get_model_categories,
    category_to_tensor,
    load_image,
    preprocess_image,
    to_array,
)
from .fgsm import attack


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

    model = load_model_default_weights(model_name=args.model)
    model.eval()  # IMPORTANT: set model to evaluation mode

    image = load_image(args.image)
    image_tensor = preprocess_image(image)

    categories = get_model_categories(model_name=args.model)

    target = category_to_tensor(args.category_truth, categories)

    new_image, old_pred, new_pred = attack(
        model, tensor=image_tensor, target=target, epsilon=args.epsilon, max_iter=int(args.max_iterations)
    )

    print(f"orig_category {args.category_truth}")
    print(f"new_category {categories[new_pred.item()]}")

    print(to_array(new_image).shape)
    if args.output is not None:
        Image.fromarray(np.uint8(255 * to_array(new_image))).save(args.output)


if __name__ == "__main__":
    main()
