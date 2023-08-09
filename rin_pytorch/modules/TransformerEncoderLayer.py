import keras_core as keras
import torch

from .DropPath import DropPath
from .MLP import MLP


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: int,
        num_heads: int,
        drop_path=0.1,
        drop_units=0.1,
        drop_att=0.0,
        self_attention=True,
        use_ffn_ln=False,
        ln_scale_shift=True,
    ):
        super().__init__()

        self.self_attention = self_attention
        if self_attention:
            self.mha_ln = keras.layers.LayerNormalization(epsilon=1e-6, center=ln_scale_shift, scale=ln_scale_shift)
            self.mha = keras.layers.MultiHeadAttention(num_heads, dim // num_heads, dropout=drop_att)

        self.mlp = MLP(
            num_layers=1,
            dim=dim,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            drop_units=drop_units,
            use_ffn_ln=use_ffn_ln,
            ln_scale_shift=ln_scale_shift,
        )

        self.dropp = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.self_attention:
            x_ln = self.mha_ln(x)
            x_residual = self.mha(x_ln, x_ln, x_ln, training=self.training)
            x = x + self.dropp(x_residual)
        x = self.mlp(x)
        return x
