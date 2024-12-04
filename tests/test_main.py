from unittest.mock import patch, MagicMock
import argparse
import pytest

from adversarial_attack.__main__ import main

from adversarial_attack import resnet_utils


@patch("adversarial_attack.__main__.preprocess_image", MagicMock(return_value="preprocessed_image"))
@patch("adversarial_attack.__main__.load_image", MagicMock(return_value="loaded_image"))
@patch("adversarial_attack.__main__.get_model_categories", MagicMock(return_value=["cat", "dog"]))
@patch("adversarial_attack.__main__.load_model_default_weights", MagicMock(return_value="loaded_model"))
@patch("argparse.ArgumentParser.parse_args")
def test_main_calls_perform_attack_with_args(mocked_args):
    # Mock arguments for standard mode
    mocked_args.return_value = argparse.Namespace(
        model="bla",
        mode="standard",
        image="path/to/image.jpg",
        category_truth="cat",
        category_target=None,
        epsilon=1.0e-3,
        max_iterations=50,
        output=None
    )

    # Patch perform_attack function
    with patch("adversarial_attack.__main__.perform_attack") as mock_perform_attack:
        # Run main
        main()

        # Assert perform_attack was called with the arguments
        mock_perform_attack.assert_called_once_with(
            mode="standard",
            model="loaded_model",
            image="preprocessed_image",
            categories=["cat", "dog"],
            true_category="cat",
            epsilon=1.0e-3,
            max_iter=50,
            target_category=None,
        )


@patch("argparse.ArgumentParser.parse_args")
def test_main_targeted_mode_missing_target(mock_args):
    # Mock arguments for targeted mode without category-target
    mock_args.return_value = argparse.Namespace(
        model="resnet18",
        mode="targeted",
        image="path/to/image.jpg",
        category_truth="cat",
        category_target=None,  # Missing target category
        epsilon=1.0e-3,
        max_iterations=50,
        output=None
    )

    # Run main and expect SystemExit due to argparse validation
    with pytest.raises(ValueError):
        main()
