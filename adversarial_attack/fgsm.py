import typing as ty
import torch
import warnings


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


def attack(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-3,
    max_iter: int = 50,
) -> ty.Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Perform a classic FGSM attack on a PyTorch model with a given image tensor.

    Args:
        model (torch.Model): PyTorch model to attack.
        tensor (torch.Tensor): Tensor to attack.
        target (torch.Tensor): Target tensor to attack.
        epsilon (float): Maximum perturbation allowed.
        max_iter (int): Maximum number of iterations to perform.

    Returns:
        torch.Tensor: Adversarial image tensor.
    """
    with torch.no_grad():
        orig_pred = model(tensor)

    if orig_pred.argmax().item() != target.item():
        raise ValueError(
            f"Model prediction {orig_pred.argmax().item()} does not match true class {target.item()}.",
            f"It is therefore pointless to perform an attack.",
        )

    # make a copy of the input tensor
    adv_tensor = tensor.clone().detach()
    orig_pred_idx = torch.tensor([orig_pred.argmax().item()])

    for i in range(max_iter):
        model.zero_grad()
        grad = compute_gradient(model=model, input=adv_tensor, target=orig_pred_idx)
        adv_tensor = torch.clamp(adv_tensor + epsilon * grad.sign(), 0, 1)
        new_pred = model(adv_tensor).argmax()
        if orig_pred_idx.item() != new_pred:
            return adv_tensor, orig_pred, new_pred

    warnings.warn(
        f"Failed to alter the prediction of the model after {max_iter} tries.",
        RuntimeWarning,
    )
    return None
