import pytest
from unittest.mock import patch, MagicMock

from adversarial_attack.__main__ import get_model_categories, load_model_default_weights


@patch("torchvision.models.ResNet18_Weights")
@patch("torchvision.models.resnet18")
def test_load_model_valid_model(mock_model, mock_weight):
    mock_weight.DEFAULT = "mocked_weights"
    model = load_model_default_weights("resnet18")
    assert model == mock_model(weights=mock_weight.DEFAULT)


def test_load_model_not_found_raises():
    with pytest.raises(ValueError):
        load_model_default_weights(model_name="not_found")


@patch("torchvision.models.ResNet18_Weights")
@patch("torchvision.models.resnet18")
def test_get_model_categories_valid_model(mock_model, mock_weight):
    mock_weight.DEFAULT = MagicMock(meta={"categories": ["dog"]})
    categories = get_model_categories("resnet18")
    assert categories == ["dog"]


def test_get_model_categories_not_found_raises():
    with pytest.raises(ValueError):
        _ = get_model_categories(model_name="not_found")