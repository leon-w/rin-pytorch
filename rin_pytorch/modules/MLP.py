import keras_core as keras
import torch

from .DropPath import DropPath
from .FeedForwardLayer import FeedForwardLayer


class MLP(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim: int,
        mlp_ratio: int,
        drop_path=0.1,
        drop_units=0.0,
        use_ffn_ln=False,
        ln_scale_shift=True,
    ):
        super().__init__()

        self.mlp_layers = torch.nn.ModuleList()
        self.layernorms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.mlp_layers.append(
                FeedForwardLayer(
                    dim,
                    dim * mlp_ratio,
                    drop_units,
                    use_ln=use_ffn_ln,
                    ln_scale_shift=ln_scale_shift,
                )
            )
            self.layernorms.append(
                keras.layers.LayerNormalization(
                    epsilon=1e-6,
                    center=ln_scale_shift,
                    scale=ln_scale_shift,
                )
            )

        self.dropp = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mlp, ln in zip(self.mlp_layers, self.layernorms):
            x_residual = self.dropp(mlp(ln(x)))
            x = x + x_residual
        return x
