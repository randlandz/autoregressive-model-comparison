import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, hidden = self.rnn(x)
        last_output = hidden[-1]

        prediction = self.linear(last_output)

        return prediction

if __name__ == "__main__":
    x = torch.randn(8, 20, 1)

    model = RNN(1, 64)

    with torch.no_grad():
        pred = model(x)

    print(x.shape)
    print(pred.shape)        
