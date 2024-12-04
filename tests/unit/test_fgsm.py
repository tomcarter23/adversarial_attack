import pytest
import torch
import torch.nn as nn
from adversarial_attack.fgsm import compute_gradient, standard_attack, targeted_attack


# Define a simple mock model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(2, 3)

    def forward(self, x):
        return torch.log_softmax(self.fc(x), dim=1)


@pytest.fixture
def model():
    model = SimpleModel()
    model.fc.weight.data = torch.tensor(
        [[1.0, -1.0, 0.5], [-1.0, 1.0, 0.5], [0.1, -0.4, 1.0]]
    )
    model.fc.bias.data = torch.tensor([0.0, 0.0, 0.0])
    return model


@pytest.fixture
def inputs_and_targets():
    inputs = torch.tensor([[1.0, 0.0, 0.5]], requires_grad=True)
    target = torch.tensor([0])
    return inputs, target


def test_compute_gradient(model, inputs_and_targets):
    inputs, target = inputs_and_targets
    grad = compute_gradient(model=model, input=inputs, target=target)

    assert grad is not None, "Gradient should not be None."
    assert grad.shape == inputs.shape, "Gradient should match input shape."


def test_standard_attack_success(model):
    tensor = torch.tensor([[1.0, 0.0, 0.5]])
    truth = torch.tensor([0])
    categories = ["cat1", "cat2", "cat3"]
    epsilon = 0.1
    max_iter = 50

    result = standard_attack(
        model=model,
        tensor=tensor,
        truth=truth,
        categories=categories,
        epsilon=epsilon,
        max_iter=max_iter,
    )

    assert result is not None, "Attack should succeed."
    adv_tensor, orig_pred, new_pred = result
    assert orig_pred.argmax() != new_pred, "Attack should change the model prediction."


def test_targeted_attack_success(model):
    tensor = torch.tensor([[1.0, 0.0, 0.5]])
    truth = torch.tensor([0])
    target = torch.tensor([2])
    categories = ["cat1", "cat2", "cat3"]
    epsilon = 0.1
    max_iter = 50

    result = targeted_attack(
        model=model,
        tensor=tensor,
        truth=truth,
        target=target,
        categories=categories,
        epsilon=epsilon,
        max_iter=max_iter,
    )

    assert result is not None, "Attack should succeed."
    adv_tensor, orig_pred, new_pred = result
    assert (
        orig_pred.argmax() != 2
    ), "Attack should change the model prediction to target."


def test_standard_attack_failure(model):
    tensor = torch.tensor([[1.0, 0.0, 0.5]])
    truth = torch.tensor([1])  # Intentionally mismatched target
    categories = ["cat1", "cat2", "cat3"]
    epsilon = 0.1
    max_iter = 50

    with pytest.warns(RuntimeWarning):
        assert (
            standard_attack(
                model=model,
                tensor=tensor,
                truth=truth,
                categories=categories,
                epsilon=epsilon,
                max_iter=max_iter,
            )
            is None
        )


def test_standard_attack_no_change(model):
    tensor = torch.tensor([[1.0, 0.0, 0.5]])
    truth = torch.tensor([0])
    categories = ["cat1", "cat2", "cat3"]
    epsilon = 1e-6  # Very small perturbation
    max_iter = 5  # Insufficient iterations to succeed

    result = standard_attack(
        model=model,
        tensor=tensor,
        truth=truth,
        categories=categories,
        epsilon=epsilon,
        max_iter=max_iter,
    )

    assert (
        result is None
    ), "Attack should fail if epsilon is too small or max_iter is insufficient."


def test_targeted_attack_no_change(model):
    tensor = torch.tensor([[1.0, 0.0, 0.5]])
    truth = torch.tensor([0])
    target = torch.tensor([2])
    categories = ["cat1", "cat2", "cat3"]

    epsilon = 1e-6  # Very small perturbation
    max_iter = 5  # Insufficient iterations to succeed

    result = targeted_attack(
        model=model,
        tensor=tensor,
        truth=truth,
        target=target,
        categories=categories,
        epsilon=epsilon,
        max_iter=max_iter,
    )

    assert (
        result is None
    ), "Attack should fail if epsilon is too small or max_iter is insufficient."
