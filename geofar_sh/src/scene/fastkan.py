import torch
from torch import nn


class GaussianRBFExpansion(nn.Module):
    """
    Lightweight fixed Gaussian basis expansion.

    This is a compact approximation used for a small appearance residual head.
    It is not intended to reproduce the full FastKAN architecture.
    """

    def __init__(self, input_dim, num_basis=8, basis_min=-1.5, basis_max=1.5, gamma=4.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_basis = int(num_basis)
        self.gamma = float(gamma)

        centers = torch.linspace(float(basis_min), float(basis_max), steps=self.num_basis, dtype=torch.float32)
        centers = centers.view(1, 1, self.num_basis).repeat(1, self.input_dim, 1)
        self.register_buffer("centers", centers)

    def forward(self, x):
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"GaussianRBFExpansion expects [N, {self.input_dim}], got {tuple(x.shape)}")
        residual = x.unsqueeze(-1) - self.centers
        expanded = torch.exp(-self.gamma * residual.pow(2))
        return expanded.reshape(x.shape[0], self.input_dim * self.num_basis)


class FastKANLayer(nn.Module):
    """
    Lightweight FastKAN-style layer:
    LayerNorm -> Gaussian basis expansion -> Linear -> SiLU.
    """

    def __init__(self, input_dim, output_dim, num_basis=8, gamma=4.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.norm = nn.LayerNorm(self.input_dim)
        self.expansion = GaussianRBFExpansion(self.input_dim, num_basis=num_basis, gamma=gamma)
        self.proj = nn.Linear(self.input_dim * int(num_basis), self.output_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.norm(x)
        x = self.expansion(x)
        x = self.proj(x)
        return self.act(x)


class FastKANResidualHead(nn.Module):
    """
    Shared-trunk residual appearance head.

    Outputs:
        raw_delta_rgb: unconstrained RGB residual logits
        raw_gate: unconstrained gate logits
    """

    def __init__(self, in_dim, hidden_dim=16, num_layers=2, num_basis=8, gamma=4.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_basis = int(num_basis)
        self.gamma = float(gamma)

        self.input_proj = nn.Linear(self.in_dim, self.hidden_dim)
        trunk = []
        for _ in range(self.num_layers):
            trunk.append(FastKANLayer(self.hidden_dim, self.hidden_dim, num_basis=self.num_basis, gamma=self.gamma))
        self.trunk = nn.ModuleList(trunk)
        self.delta_head = nn.Linear(self.hidden_dim, 3)
        self.gate_head = nn.Linear(self.hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.delta_head.weight, gain=0.1)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.xavier_uniform_(self.gate_head.weight, gain=0.1)
        nn.init.constant_(self.gate_head.bias, -2.0)

    def forward(self, x):
        if x.ndim != 2 or x.shape[-1] != self.in_dim:
            raise ValueError(f"FastKANResidualHead expects [N, {self.in_dim}], got {tuple(x.shape)}")
        hidden = torch.nn.functional.silu(self.input_proj(x))
        for layer in self.trunk:
            hidden = hidden + layer(hidden)
        raw_delta = self.delta_head(hidden)
        raw_gate = self.gate_head(hidden)
        return raw_delta, raw_gate



