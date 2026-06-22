from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only in environments without torch
    torch = None
    nn = None


def require_torch():
    if torch is None or nn is None:
        raise ImportError(
            "PyTorch is required for hyb_nurbs.ml. Install the ML extra, for example: "
            "python -m pip install -e .[ml]"
        )
    return torch


if nn is not None:

    class MLPRegressor(nn.Module):
        """Configurable MLP for NURBS control-point regression."""

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: list[int] | tuple[int, ...] = (64, 128, 128, 64),
            activation: str = "relu",
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev = input_dim
            for dim in hidden_dims:
                layers.append(nn.Linear(prev, int(dim)))
                layers.append(_activation(activation))
                if dropout > 0:
                    layers.append(nn.Dropout(float(dropout)))
                prev = int(dim)
            layers.append(nn.Linear(prev, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x)


    def _activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        raise ValueError(f"Unsupported activation: {name}")

else:

    class MLPRegressor:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401, ANN002, ANN003
            require_torch()
