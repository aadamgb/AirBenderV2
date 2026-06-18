import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            is_last = (i == len(layer_sizes) - 2)
            layers.append(output_activation() if is_last else activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)