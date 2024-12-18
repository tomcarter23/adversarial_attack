import typing as ty
import torch
import warnings
import logging

from .resnet_utils import category_to_tensor

logger = logging.getLogger("adversarial_attack")


def compute_gradient(model: torch.nn.Module, input: torch.Tensor, target: torch.Tensor):
    """
    Compute the gradients of the input tensor with respect to the target class.

    Args:
        model (torch.Model): PyTorch model to compute gradients for.
        input (torch.Tensor): Input tensor to compute gradients for.
        target (torch.Tensor): Target tensor to compute gradients for.

    Returns:
        torch.Tensor: Gradients of the input tensor with respect to the target class.
    """
    input.requires_grad = True

    output = model(input)

    loss = torch.nn.functional.nll_loss(output, target=target)

    loss.backward()

    input.requires_grad = False

    return input.grad.data


def standard_attack(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    truth: torch.Tensor,
    epsilon: float = 1e-3,
    max_iter: int = 50,
) -> ty.Optional[tuple[torch.Tensor, int, int]]:
    """
    Perform a classic FGSM attack on a PyTorch model with a given image tensor.

    Args:
        model (torch.Model): PyTorch model to attack.
        tensor (torch.Tensor): Tensor to attack.
        truth (torch.Tensor): Tensor representing true category.
        epsilon (float): Maximum perturbation allowed.
        max_iter (int): Maximum number of iterations to perform.

    Returns:
        torch.Tensor: Adversarial image tensor or None if attack failed.
    """
    logger.info("Conducting standard attack")

    with torch.no_grad():
        orig_pred = model(tensor)
    logger.debug(
        f"Original prediction class: {orig_pred.argmax()}, probability: {torch.nn.functional.softmax(orig_pred, dim=1).max()}"
    )

    orig_pred_idx: int = orig_pred.argmax().item()
    truth_idx: int = truth.item()

    if orig_pred_idx != truth_idx:
        warnings.warn(
            (
                f"Model prediction {orig_pred_idx} does not match true class {truth_idx}."
                f"It is therefore pointless to perform an attack."
            ),
        )
        return None

    # make a copy of the input tensor
    adv_tensor = tensor.clone().detach()

    for i in range(max_iter):
        model.zero_grad()
        grad = compute_gradient(model=model, input=adv_tensor, target=torch.tensor([orig_pred_idx]))
        adv_tensor = torch.clamp(adv_tensor + epsilon * grad.sign(), -2, 2)
        new_output = model(adv_tensor)
        new_pred_idx = new_output.argmax()
        logger.debug(
            f"attack iteration {i}, current prediction: {new_pred_idx}, current max probability: {torch.nn.functional.softmax(new_output, dim=1).max()}"
        )
        if orig_pred_idx != new_pred_idx:
            logger.info(f"Standard attack successful.")
            return adv_tensor, orig_pred_idx, new_pred_idx

    logger.warning(
        f"Failed to alter the prediction of the model after {max_iter} tries.",
    )
    return None


def targeted_attack(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    truth: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-3,
    max_iter: int = 50,
) -> ty.Optional[tuple[torch.Tensor, int, int]]:
    """
    Perform a targeted FGSM attack on a PyTorch model with a given image tensor.

    Args:
        model (torch.Model): PyTorch model to attack.
        tensor (torch.Tensor): Tensor to attack.
        truth (torch.Tensor): Tensor representing true category.
        target (torch.Tensor): Tensor representing targeted category.
        epsilon (float): Maximum perturbation allowed.
        max_iter (int): Maximum number of iterations to perform.

    Returns:
        torch.Tensor: Adversarial image tensor or None if attack failed.
    """
    logger.info("Conducting targeted attack")

    with torch.no_grad():
        orig_pred = model(tensor)
    logger.debug(
        f"Original prediction class: {orig_pred.argmax()}, probability: {torch.nn.functional.softmax(orig_pred, dim=1).max()}"
    )

    orig_pred_idx: int = orig_pred.argmax().item()
    truth_idx: int = truth.item()
    target_idx: int = target.item()

    if orig_pred_idx != truth_idx:
        logger.warning(
            (
                f"Model prediction {orig_pred_idx} does not match true class {truth_idx}."
                f"It is therefore pointless to perform an attack.",
            )
        )
        return None

    # make a copy of the input tensor
    adv_tensor = tensor.clone().detach()

    for i in range(max_iter):
        model.zero_grad()
        grad = compute_gradient(model=model, input=adv_tensor, target=target)
        adv_tensor = torch.clamp(adv_tensor - epsilon * grad.sign(), -2, 2)
        new_output = model(adv_tensor)
        new_pred_idx = new_output.argmax(dim=1).item()
        logger.debug(
            f"Attack iteration {i}, target: {target_idx}, current prediction: {new_pred_idx}, current max probability: {torch.nn.functional.softmax(new_output, dim=1).max()}"
        )
        if new_pred_idx == target_idx:
            logger.info(f"Targeted attack successful.")
            return adv_tensor, orig_pred_idx, new_pred_idx

    logger.warning(
        f"Failed to achieve target prediction of the model after {max_iter} tries.",
    )
    return None


def get_attack_fn(
    mode: str,
    model: torch.nn.Module,
    tensor: torch.Tensor,
    category_truth: str,
    category_target: str,
    categories: list[str],
    epsilon: float,
    max_iter: int,
):
    def execute():
        # Choose the correct attack function based on mode
        truth = category_to_tensor(category_truth, categories)
        if mode == "standard":
            attack_method = standard_attack
            return attack_method(
                model,
                tensor=tensor,
                truth=truth,
                epsilon=epsilon,
                max_iter=max_iter,
            )
        elif mode == "targeted":
            target = category_to_tensor(category_target, categories)
            attack_method = targeted_attack
            if target is None:
                raise ValueError("Target must be specified for targeted attack.")

            return attack_method(
                model,
                tensor=tensor,
                truth=truth,
                target=target,
                epsilon=epsilon,
                max_iter=max_iter,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return execute
