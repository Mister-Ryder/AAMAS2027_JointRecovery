"""Small typed low-/high-pass action-ranking network, matching native C++ inference.

This is NOT a reproduction of GCON, not a universal CO neural solver, and not a
learned pruning certificate. layers=0 is the no-message-passing MLP control.
"""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn

FEATURE_DIM, CONTEXT_DIM = 10, 6

class RankingNet(nn.Module):
    def __init__(self, hidden: int = 16, layers: int = 2):
        super().__init__()
        if not 1 <= hidden <= 64 or not 0 <= layers <= 4:
            raise ValueError("native-supported range: hidden 1..64, layers 0..4")
        self.hidden, self.depth = hidden, layers
        self.input = nn.Linear(FEATURE_DIM, hidden)
        self.layers = nn.ModuleList([nn.ModuleDict({
            "self": nn.Linear(hidden, hidden),
            "backbone": nn.Linear(hidden, hidden, bias=False),
            "cross": nn.Linear(hidden, hidden, bias=False),
            "high": nn.Linear(hidden, hidden, bias=False)}) for _ in range(layers)])
        self.head = nn.Linear(2 * hidden + CONTEXT_DIM, hidden)
        self.out = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor, edges: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != FEATURE_DIM or len(x) == 0:
            raise ValueError("nonempty node features of shape (n,10) required")
        h = torch.relu(self.input(x))
        for layer in self.layers:
            messages, degrees = [], []
            for relation in (0, 1):
                pair = edges[edges[:, 2] == relation, :2]
                src = torch.cat((pair[:, 0], pair[:, 1]))
                dst = torch.cat((pair[:, 1], pair[:, 0]))
                aggregate = torch.zeros_like(h).index_add(0, dst, h[src])
                degree = torch.zeros((len(h), 1), dtype=h.dtype, device=h.device)
                degree.index_add_(0, dst, torch.ones((len(dst), 1), dtype=h.dtype, device=h.device))
                messages.append(aggregate); degrees.append(degree)
            high = h - (messages[0] + messages[1]) / (degrees[0] + degrees[1]).clamp(min=1)
            h = torch.relu(layer["self"](h)
                           + layer["backbone"](messages[0] / degrees[0].clamp(min=1))
                           + layer["cross"](messages[1] / degrees[1].clamp(min=1))
                           + layer["high"](high))
        outsider = x[:, 3] > .5
        outside_pool = h[outsider].sum(dim=0) / outsider.sum().clamp(min=1)
        pooled = torch.cat((h.mean(dim=0), outside_pool, context))
        return self.out(torch.relu(self.head(pooled)))

    def action(self, action: dict) -> torch.Tensor:
        parameter = next(self.parameters())
        x = torch.as_tensor(action["x"], dtype=parameter.dtype, device=parameter.device)
        edges = torch.as_tensor(action["edges"], dtype=torch.long, device=parameter.device).reshape(-1, 3)
        context = torch.as_tensor(action["context"], dtype=parameter.dtype, device=parameter.device)
        return self(x, edges, context)

    def export(self, path: Path) -> None:
        """Text only: inference does not load pickle or require PyTorch."""
        ordered = [self.input.weight, self.input.bias]
        for layer in self.layers:
            ordered.extend((layer["self"].weight, layer["self"].bias,
                            layer["backbone"].weight, layer["cross"].weight, layer["high"].weight))
        ordered.extend((self.head.weight, self.head.bias, self.out.weight, self.out.bias))
        with Path(path).open("w", encoding="ascii", newline="\n") as f:
            f.write(f"BARRNN1 {FEATURE_DIM} {self.hidden} {self.depth} {CONTEXT_DIM}\n")
            for tensor in ordered:
                f.write(" ".join(format(float(x), ".17g") for x in tensor.detach().cpu().reshape(-1)) + "\n")


def native_score(prediction, scale: float) -> float:
    import math
    return math.expm1(max(0., min(20., float(prediction[0])))) * scale / math.exp(max(-16., min(10., float(prediction[1]))))
