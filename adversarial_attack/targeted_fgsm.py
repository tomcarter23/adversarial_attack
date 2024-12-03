import torch


def attack(
    tensor: torch.Tensor, model: torch.Model, epsilon: float = 1e-3, max_iter: int = 50
) -> torch.Tensor:
    """

    :param tensor:
    :param model:
    :param epsilon:
    :param max_iter:
    :return:
    """
