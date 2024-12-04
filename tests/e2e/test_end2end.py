import pytest

from adversarial_attack.api import perform_attack
from adversarial_attack.resnet_utils import load_model_default_weights, preprocess_image, get_model_categories, load_image

# Global variables to track test results
success_count = 0
total_tests = 0


@pytest.mark.parametrize(
    "image_truth",
    [
        ("./tests/e2e/input/beaker_ILSVRC2012_val_00001780.JPEG", "beaker"),
        ("./tests/e2e/input/bookcase_ILSVRC2012_val_00031142.JPEG", "bookcase"),
        ("./tests/e2e/input/doormat_ILSVRC2012_val_00030383.JPEG", "doormat"),
        ("./tests/e2e/input/hare_ILSVRC2012_val_00004064.JPEG", "hare"),
        ("./tests/e2e/input/jack-o'-lantern_ILSVRC2012_val_00030955.JPEG", "jack-o'-lantern"),
        ("./tests/e2e/input/lawn_mower_ILSVRC2012_val_00020327.JPEG", "lawn mower"),
        ("./tests/e2e/input/lionfish_ILSVRC2012_val_00019791.JPEG", "lionfish"),
        ("./tests/e2e/input/monarch_ILSVRC2012_val_00002935.JPEG", "monarch"),
        ("./tests/e2e/input/pickelhaube_ILSVRC2012_val_00018444.JPEG", "pickelhaube"),
        ("./tests/e2e/input/sea_urchin_ILSVRC2012_val_00028454.JPEG", "sea urchin"),
    ],
    ids=[
        "input: beaker", "input: bookcase", "input: doormat", "input: hare",
        "input: jack-o'-lantern", "input: lawn_mower", "input: lionfish",
        "input: monarch", "input: pickelhaube", "input: sea_urchin",
    ]
)
@pytest.mark.parametrize("model_name", ["resnet50", "resnet101", "resnet152"], ids=["model: resnet50", "model: resnet101", "model: resnet152"])
def test_perform_attack_standard(model_name, image_truth):
    global success_count, total_tests
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
        epsilon=1.0e-1,
        max_iter=10,
    )

    total_tests += 1
    if result is not None:
        success_count += 1

    assert result is not None


@pytest.mark.parametrize(
    "image_truth",
    [
        ("./tests/e2e/input/beaker_ILSVRC2012_val_00001780.JPEG", "beaker"),
        ("./tests/e2e/input/bookcase_ILSVRC2012_val_00031142.JPEG", "bookcase"),
        ("./tests/e2e/input/doormat_ILSVRC2012_val_00030383.JPEG", "doormat"),
        ("./tests/e2e/input/hare_ILSVRC2012_val_00004064.JPEG", "hare"),
        ("./tests/e2e/input/jack-o'-lantern_ILSVRC2012_val_00030955.JPEG", "jack-o'-lantern"),
        ("./tests/e2e/input/lawn_mower_ILSVRC2012_val_00020327.JPEG", "lawn mower"),
        ("./tests/e2e/input/lionfish_ILSVRC2012_val_00019791.JPEG", "lionfish"),
        ("./tests/e2e/input/monarch_ILSVRC2012_val_00002935.JPEG", "monarch"),
        ("./tests/e2e/input/pickelhaube_ILSVRC2012_val_00018444.JPEG", "pickelhaube"),
        ("./tests/e2e/input/sea_urchin_ILSVRC2012_val_00028454.JPEG", "sea urchin"),
    ],
    ids=[
        "input: beaker", "input: bookcase", "input: doormat", "input: hare",
        "input: jack-o'-lantern", "input: lawn_mower", "input: lionfish",
        "input: monarch", "input: pickelhaube", "input: sea_urchin",
    ]
)
@pytest.mark.parametrize(
    "target_category",
    [
        "beaker",
        "bookcase",
        "doormat",
        "hare",
        "jack-o'-lantern",
        "lawn mower",
        "lionfish",
        "monarch",
        "pickelhaube",
        "sea urchin",
    ]
)
@pytest.mark.parametrize("model_name", ["resnet50", "resnet101", "resnet152"])
def test_perform_attack_targeted(model_name, target_category, image_truth):
    global success_count, total_tests
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
        epsilon=1.0e-1,
        max_iter=10,
    )

    total_tests += 1
    if result is not None:
        success_count += 1

    assert result is not None


def pytest_sessionfinish(request):
    def session_finish():
        success_rate = success_count / total_tests if total_tests > 0 else 0
        if success_rate < 0.75:
            pytest.exit(f"Test success rate {success_rate * 100:.2f}% is below the expected 75%", returncode=1)

    request.addfinalizer(session_finish)
