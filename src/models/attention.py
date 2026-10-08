import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, input_size=1, window_size=20, hidden_size=64, num_heads=4):
        super().__init__()

        self.input_proj = nn.Linear(input_size, hidden_size)

        self.position_emb = nn.Parameter(
            torch.zeros(1, window_size, hidden_size)
        )

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        self.attention_output = nn.Linear(hidden_size, hidden_size)

        self.output = nn.Linear(hidden_size, 1)

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        x = self.input_proj(x)
        x = x + self.position_emb[:, :seq_len, :]

        Q = self.q_proj(x).reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )
        K = self.k_proj(x).reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )
        V = self.v_proj(x).reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = (Q @ K.transpose(-1, -2)) / (self.head_dim ** 0.5)

        attention_weights = torch.softmax(scores, dim=-1)

        attention = attention_weights @ V
        attention = attention.transpose(1, 2).reshape(
            batch_size,
            seq_len,
            self.num_heads * self.head_dim
        )
        attention = self.attention_output(attention)

        prediction = self.output(attention[:, -1, :])

        return prediction

if __name__ == "__main__":
    x = torch.randn(8, 20, 1)

    model = Attention()

    with torch.no_grad():
        prediction = model(x)

    print(prediction.shape)
