import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, window_size=20, hidden_size=64):
        super().__init__()

        self.flatten = nn.Flatten(start_dim=1)

        self.fc1 = nn.Linear(window_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.output(x)

        return x


if __name__ == "__main__":
    x = torch.randn([8, 20, 1])

    model = MLP()

    with torch.no_grad():
        output = model(x)

    print(
        x.shape,
        output.shape
    )

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )