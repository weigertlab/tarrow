import torch


class DecorrelationLoss(torch.nn.Module):
    """Penalizes correlated features on a single image level."""

    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, x):
        assert x.ndim == 4
        x = x.flatten(2, 3)
        # x =  torch.nn.functional.normalize(x, dim=1)
        dot = (x @ x.transpose(2, 1)) / self.temperature
        logits = torch.softmax(dot, dim=-1)
        loss = -torch.log(torch.diagonal(logits, dim1=-2, dim2=-1) + 1e-10)
        loss = torch.mean(loss)

        return loss
