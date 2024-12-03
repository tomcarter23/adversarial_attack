import pytest
import torch
import torch.nn as nn
from adversarial_attack.fgsm import compute_gradient, attack


# Define a simple mock model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        return torch.log_softmax(self.fc(x), dim=1)


@pytest.fixture
def model():
    model = SimpleModel()
    model.fc.weight.data = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    model.fc.bias.data = torch.tensor([0.0, 0.0])
    return model


@pytest.fixture
def inputs_and_targets():
    inputs = torch.tensor([[1.0, 0.0]], requires_grad=True)
    target = torch.tensor([0])
    return inputs, target


def test_compute_gradient(model, inputs_and_targets):
    inputs, target = inputs_and_targets
    grad = compute_gradient(model=model, input=inputs, target=target)

    assert grad is not None, "Gradient should not be None."
    assert grad.shape == inputs.shape, "Gradient should match input shape."


def test_attack_success(model):
    tensor = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([0])
    epsilon = 0.1
    max_iter = 50

    result = attack(model=model, tensor=tensor, target=target, epsilon=epsilon, max_iter=max_iter)

    assert result is not None, "Attack should succeed."
    adv_tensor, orig_pred, new_pred = result
    assert orig_pred.argmax() != new_pred, "Attack should change the model prediction."


def test_attack_failure(model):
    tensor = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([1])  # Intentionally mismatched target
    epsilon = 0.1
    max_iter = 50

    with pytest.raises(ValueError, match="does not match true class"):
        attack(model=model, tensor=tensor, target=target, epsilon=epsilon, max_iter=max_iter)


def test_attack_no_change(model):
    tensor = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([0])
    epsilon = 1e-6  # Very small perturbation
    max_iter = 5  # Insufficient iterations to succeed

    result = attack(model=model, tensor=tensor, target=target, epsilon=epsilon, max_iter=max_iter)

    assert result is None, "Attack should fail if epsilon is too small or max_iter is insufficient."
