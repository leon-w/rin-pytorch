import keras_core as keras
import torch


class FeedForwardLayer(torch.nn.Module):
    def __init__(
        self,
        dim_att: int,
        dim_mlp: int,
        drop_units=0.1,
        use_ln=False,
        ln_scale_shift=False,
    ):
        super().__init__()
        self.dense1 = keras.layers.Dense(dim_mlp, activation="gelu")
        self.dropout = keras.layers.Dropout(drop_units)
        self.dense2 = keras.layers.Dense(dim_att)
        if use_ln:
            self.ln = keras.layers.LayerNormalization(
                epsilon=1e-6,
                center=ln_scale_shift,
                scale=ln_scale_shift,
            )
        else:
            self.ln = keras.layers.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = self.ln(x)
        x = self.dropout(x, training=self.training)
        x = self.dense2(x)
        return x
