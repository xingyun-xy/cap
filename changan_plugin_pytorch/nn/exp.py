import torch


class Exp(torch.nn.Module):
    """Module implementation of torch.exp"""

    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, input):
        return torch.exp(input)
