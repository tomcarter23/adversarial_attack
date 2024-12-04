import pytest
from adversarial_attack.api import perform_attack
from adversarial_attack.resnet_utils import load_model_default_weights, preprocess_image, get_model_categories, load_image


@pytest.fixture
def image_truth():
    return "./tests/e2e/input/lionfish_ILSVRC2012_val_00019791.JPEG", "lionfish"


@pytest.mark.parametrize("model_name", ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
def test_perform_attack_standard(model_name, image_truth):
    image, true_category = image_truth
    model = load_model_default_weights(model_name)
    input_image = preprocess_image(load_image(image))
    categories = get_model_categories(model_name)
    result = perform_attack(
        model=model,
        mode="standard",
        image=input_image,
        categories=categories,
        true_category=true_category,
        epsilon=1.0e-3,
        max_iter=50,
    )

    assert result is not None


@pytest.mark.parametrize("target_category", ["goldfish", "monarch"])
@pytest.mark.parametrize("model_name", ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
def test_perform_attack_standard(model_name, target_category, image_truth):
    image, true_category = image_truth
    model = load_model_default_weights(model_name)
    input_image = preprocess_image(load_image(image))
    categories = get_model_categories(model_name)
    result = perform_attack(
        model=model,
        mode="targeted",
        image=input_image,
        categories=categories,
        true_category=true_category,
        target_category=target_category,
        epsilon=1.0e-3,
        max_iter=50,
    )

    assert result is not None
