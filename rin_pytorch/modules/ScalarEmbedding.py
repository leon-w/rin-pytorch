import keras_core as keras
import torch

from ..utils.pos_embedding import positional_encoding


class ScalarEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        scaling: float | torch.Tensor,
        expansion=4,
    ):
        super().__init__()
        self.scalar_encoding = lambda x: positional_encoding(x * scaling, dim)
        initializer = keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        self.dense_0 = keras.layers.Dense(dim * expansion, kernel_initializer=initializer)
        self.dense_1 = keras.layers.Dense(dim * expansion, kernel_initializer=initializer)

    def forward(
        self,
        x: torch.Tensor,
        last_swish=True,
        normalize=False,
    ) -> torch.Tensor:
        assert x.ndim == 1
        x = self.scalar_encoding(x)[0]
        if normalize:
            x_mean = torch.mean(x, -1, keepdim=True)
            x_std = torch.std(x, -1, correction=0, keepdim=True)
            x = (x - x_mean) / x_std
        x = keras.activations.silu(self.dense_0(x))
        x = self.dense_1(x)
        return keras.activations.silu(x) if last_swish else x
