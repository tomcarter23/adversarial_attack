import pytest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from PIL import Image

from adversarial_attack.resnet_utils import (
    load_image,
    preprocess_image,
    to_array,
    category_to_tensor,
    get_model_categories,
    load_model_default_weights,
)


@pytest.fixture
def sample_image_path(tmp_path):
    # Create a temporary sample image for testing
    image = Image.new("RGB", (300, 300), color="red")
    sample_path = tmp_path / "sample.jpg"
    image.save(sample_path)
    return str(sample_path)


@pytest.fixture
def sample_image(sample_image_path):
    return load_image(sample_image_path)


@pytest.fixture
def categories():
    return ["cat", "dog", "bird"]


def test_load_image(sample_image_path):
    # Test image loading functionality
    image = load_image(sample_image_path)
    assert isinstance(image, Image.Image), "Loaded object should be a PIL Image"
    assert image.size == (300, 300), "Image size should match the sample image"


def test_preprocess_image(sample_image):
    # Test preprocessing functionality
    preprocessed = preprocess_image(sample_image)
    assert isinstance(
        preprocessed, torch.Tensor
    ), "Preprocessed output should be a tensor"
    assert preprocessed.shape == (
        1,
        3,
        224,
        224,
    ), "Tensor shape should match ResNet input requirements"


def test_to_array(sample_image):
    # Test conversion from tensor to numpy array
    preprocessed = preprocess_image(sample_image)
    array = to_array(preprocessed)
    assert isinstance(array, np.ndarray), "Output should be a numpy array"
    assert array.shape == (
        224,
        224,
        3,
    ), "Array shape should match the expected image shape after preprocessing"


def test_category_to_tensor(categories):
    # Test conversion of category to tensor
    category = "dog"
    category_tensor = category_to_tensor(category, categories)
    assert isinstance(category_tensor, torch.Tensor), "Output should be a tensor"
    assert (
        category_tensor.item() == 1
    ), "Tensor value should correspond to the index of the category"


def test_invalid_category_to_tensor_raises(categories):
    # Test invalid category
    with pytest.raises(ValueError):
        category_to_tensor("elephant", categories)


def test_preprocess_to_array_consistency(sample_image):
    # Check the consistency between preprocess_image and to_array
    preprocessed = preprocess_image(sample_image)
    array = to_array(preprocessed)
    assert np.allclose(
        array[100, 100], np.array([255, 0, 0]) / 255, atol=0.1
    ), "Center pixel value should approximately match the original image color (normalized)"


@patch("torchvision.models.ResNet50_Weights")
@patch("torchvision.models.resnet50")
def test_load_model_valid_model(mock_model, mock_weight):
    mock_weight.DEFAULT = "mocked_weights"
    model = load_model_default_weights("resnet50")
    assert model == mock_model(weights=mock_weight.DEFAULT)


def test_load_model_not_found_raises():
    with pytest.raises(ValueError):
        load_model_default_weights(model_name="not_found")


@patch("torchvision.models.ResNet50_Weights")
@patch("torchvision.models.resnet50")
def test_get_model_categories_valid_model(mock_model, mock_weight):
    mock_weight.DEFAULT = MagicMock(meta={"categories": ["dog"]})
    categories = get_model_categories("resnet50")
    assert categories == ["dog"]


def test_get_model_categories_not_found_raises():
    with pytest.raises(ValueError):
        _ = get_model_categories(model_name="not_found")
