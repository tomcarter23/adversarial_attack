import pytest

from adversarial_attack.api import perform_attack
from adversarial_attack.resnet_utils import load_model_default_weights, preprocess_image, get_model_categories, load_image

import pytest
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_perform_attack_standard():
    test_cases = [
        ("./tests/e2e/input/beaker_ILSVRC2012_val_00001780.JPEG", "beaker"),
        ("./tests/e2e/input/doormat_ILSVRC2012_val_00030383.JPEG", "doormat"),
        ("./tests/e2e/input/hare_ILSVRC2012_val_00004064.JPEG", "hare"),
        ("./tests/e2e/input/jack-o'-lantern_ILSVRC2012_val_00030955.JPEG", "jack-o'-lantern"),
        ("./tests/e2e/input/lawn_mower_ILSVRC2012_val_00020327.JPEG", "lawn mower"),
        ("./tests/e2e/input/lionfish_ILSVRC2012_val_00019791.JPEG", "lionfish"),
        ("./tests/e2e/input/monarch_ILSVRC2012_val_00002935.JPEG", "monarch"),
        ("./tests/e2e/input/pickelhaube_ILSVRC2012_val_00018444.JPEG", "pickelhaube"),
        ("./tests/e2e/input/sea_urchin_ILSVRC2012_val_00028454.JPEG", "sea urchin"),
    ]

    models = ["resnet50", "resnet101", "resnet152"]

    total_tests = 0
    success_count = 0

    logger.info("Starting test for perform_attack_standard...")

    for model_name in models:
        logger.info(f"Testing model: {model_name}")
        for image_path, true_category in test_cases:
            try:
                logger.info(f"Running test for image '{image_path}' with true category '{true_category}'")

                model = load_model_default_weights(model_name)
                input_image = preprocess_image(load_image(image_path))
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
                else:
                    logger.warning(
                        f"Test failed for model '{model_name}', image '{image_path}', true category '{true_category}'")

            except Exception as e:
                logger.error(
                    f"Error occurred for model '{model_name}', image '{image_path}', true category '{true_category}': {e}")
                total_tests += 1  # Count this as a test to avoid skewing success rate

    success_rate = success_count / total_tests if total_tests > 0 else 0
    logger.info(f"Completed all tests. Success rate: {success_rate:.2%} ({success_count}/{total_tests})")
    assert success_rate >= 0.75, f"Success rate {success_rate:.2%} is below the required threshold of 75%."


def test_perform_attack_targeted():
    test_cases = [
        ("./tests/e2e/input/beaker_ILSVRC2012_val_00001780.JPEG", "beaker"),
        ("./tests/e2e/input/doormat_ILSVRC2012_val_00030383.JPEG", "doormat"),
        ("./tests/e2e/input/hare_ILSVRC2012_val_00004064.JPEG", "hare"),
        ("./tests/e2e/input/jack-o'-lantern_ILSVRC2012_val_00030955.JPEG", "jack-o'-lantern"),
        ("./tests/e2e/input/lawn_mower_ILSVRC2012_val_00020327.JPEG", "lawn mower"),
        ("./tests/e2e/input/lionfish_ILSVRC2012_val_00019791.JPEG", "lionfish"),
        ("./tests/e2e/input/monarch_ILSVRC2012_val_00002935.JPEG", "monarch"),
        ("./tests/e2e/input/pickelhaube_ILSVRC2012_val_00018444.JPEG", "pickelhaube"),
        ("./tests/e2e/input/sea_urchin_ILSVRC2012_val_00028454.JPEG", "sea urchin"),
    ]

    target_categories = [
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

    models = ["resnet50", "resnet101", "resnet152"]

    total_tests = 0
    success_count = 0

    logger.info("Starting test for perform_attack_targeted...")

    for model_name in models:
        logger.info(f"Testing model: {model_name}")
        for target_category in target_categories:
            logger.info(f"Testing target category: {target_category}")
            for image_path, true_category in test_cases:
                try:
                    logger.info(
                        f"Running test for image '{image_path}' with true category '{true_category}' targeting '{target_category}'")

                    model = load_model_default_weights(model_name)
                    input_image = preprocess_image(load_image(image_path))
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
                    else:
                        logger.warning(f"Test failed for model '{model_name}', image '{image_path}', "
                                       f"true category '{true_category}', targeting '{target_category}'")

                except Exception as e:
                    logger.error(f"Error occurred for model '{model_name}', image '{image_path}', "
                                 f"true category '{true_category}', targeting '{target_category}': {e}")
                    total_tests += 1  # Count this as a test to avoid skewing success rate

    success_rate = success_count / total_tests if total_tests > 0 else 0
    logger.info(f"Completed all tests. Success rate: {success_rate:.2%} ({success_count}/{total_tests})")
    assert success_rate >= 0.75, f"Success rate {success_rate:.2%} is below the required threshold of 75%."
