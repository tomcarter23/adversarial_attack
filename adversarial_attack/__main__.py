import argparse
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


def get_model_categories(model_name) -> list[str]:
    """
    Get the categories of models available.

    Returns:
        list[str]: List of model categories.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found.")

    return getattr(models, AVAILABLE_MODELS[model_name]).DEFAULT.meta["categories"]


def main():
    parser = argparse.ArgumentParser(
        description="Run adversarial attack on a PyTorch model with a given image."
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Path to the PyTorch model to attack.",
        required=True,
    )
    parser.add_argument(
        "--image",
        "-i",
        help="Path to the image to attack.",
        required=False,
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to save the adversarial image.",
        required=False,
    )
    parser.add_argument(
        "--repo",
        help="Repo to load model from.",
        default="pytorch/vision",
        required=False,
    )

    args = parser.parse_args()

    model = load_model_default_weights(model_name=args.model)
    print(get_model_categories(model_name=args.model))


if __name__ == "__main__":
    main()