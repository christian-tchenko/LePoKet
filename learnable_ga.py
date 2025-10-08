import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableGeneticAttention(nn.Module):
    """
    Learnable Genetic Attention.
    Learns a lambda ∈ [0,1] initialized randomly.
    """
    def __init__(self, channels, hidden_dim=None, learn_lambda=True):
        super().__init__()
        hidden_dim = hidden_dim or channels

        self.Wq = nn.Linear(channels, hidden_dim, bias=False)
        self.Wk = nn.Linear(channels, hidden_dim, bias=False)
        self.Wv = nn.Linear(channels, hidden_dim, bias=False)

        if learn_lambda:
            lambda_init = torch.rand(1)  # random initialization of λ ∈ [0,1]
            self.vlambda = nn.Parameter(lambda_init)
        else:
            self.register_buffer("vlambda", torch.tensor(0.6))

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        for layer in [self.Wq, self.Wk, self.Wv]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")

    def forward(self, query, key, value):
        # All expected as [B, C]
        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)

        # Basic similarity attention
        score = torch.sum(Q * K, dim=1, keepdim=True) / (Q.size(1) ** 0.5)
        attn = torch.sigmoid(score)
        gate = self.gate(Q * K)
        attn = 0.5 * (attn + gate)

        # Learnable λ fusion - Return Mixtured output
        fused = query + self.vlambda * (attn * V + (1 - attn) * query - query)
        return fused
