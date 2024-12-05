import argparse
import sys
from PIL import Image
import numpy as np
import logging

from adversarial_attack.resnet_utils import (
    AVAILABLE_MODELS,
    load_model_default_weights,
    get_model_categories,
    load_image,
    to_array,
    preprocess_image,
)
from adversarial_attack.api import perform_attack

logger = logging.getLogger("adversarial_attack")


def setup_logging(level: str):
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s\t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)


def main():
    """
    CLI entry point for the adversarial attack.
    Can perform a targeted or standard attack on a given image for a torchvision resnet model
    supported by the software.

    Usage examples:
        $ python -m adversarial_attack --model resnet18 --mode targeted --image path/to/image.jpg --category-truth cat \
            --category-target dog --epsilon 1.0e-3 --max-iterations 50 --output path/to/output.jpg

        $ python -m adversarial_attack --model resnet18 --mode standard --image path/to/image.jpg --category-truth cat \
             --epsilon 1.0e-3 --max-iterations 50 --output path/to/output.jpg
    """

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
    parser.add_argument(
        "--log",
        default="WARNING",
        help=f"Set the logging level. Available options: {list(logging._nameToLevel.keys())}",
    )

    args = parser.parse_args()

    setup_logging(level=args.log)

    if args.mode == "targeted" and args.category_target is None:
        raise ValueError("Target category is required for targeted attacks.")

    model = load_model_default_weights(model_name=args.model)

    image = load_image(args.image)
    image_tensor = preprocess_image(image)

    out_image = perform_attack(
        mode=args.mode,
        model=model,
        image=image_tensor,
        categories=get_model_categories(args.model),
        true_category=args.category_truth,
        epsilon=float(args.epsilon),
        max_iter=int(args.max_iterations),
        target_category=args.category_target,
    )

    if out_image is None:
        logger.info("No adversarial generated. If output requested no image will be saved.")
        return None

    if args.output is not None:
        logger.info(f"Saving adversarial image to {args.output}")
        Image.fromarray(np.uint8(255 * to_array(out_image))).save(args.output)

    else:
        logger.info("No output image to save.")


if __name__ == "__main__":
    main()
