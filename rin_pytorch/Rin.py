import keras_core as keras
import numpy as np
import torch
from einops import rearrange

from .modules import MLP, LambdaModule, ScalarEmbedding, TransformerDecoderLayer, TransformerEncoder
from .utils.pos_embedding import create_2d_sin_cos_pos_emb


def _concat_tokens(*tokens: torch.Tensor | None) -> torch.Tensor:
    # tokens in shape [..., n, d]
    return torch.cat([t for t in tokens if t is not None], -2)


class Rin(torch.nn.Module):
    def __init__(
        self,
        num_layers: str,
        latent_slots: int,
        latent_dim: int,
        latent_mlp_ratio: int,
        latent_num_heads: int,
        tape_dim: int,
        tape_mlp_ratio: int,
        rw_num_heads: int,
        image_height: int,
        image_width: int,
        image_channels: int,
        patch_size: int,
        latent_pos_encoding="learned",
        tape_pos_encoding="learned",
        drop_path=0.0,
        drop_units=0.1,
        drop_att=0.0,
        time_scaling=1e4,
        self_cond="none",
        time_on_latent=False,
        cond_on_latent_n=0,
        cond_tape_writable=False,
        cond_dim=0,
        cond_proj=True,
        cond_decoupled_read=False,
        xattn_enc_ln=False,
    ):
        super().__init__()

        self._image_height = image_height
        self._image_width = image_width
        self._image_channels = image_channels
        self._n_rows = image_height // patch_size
        self._n_cols = image_width // patch_size
        self._num_tokens = self._n_rows * self._n_cols
        self._patch_size = patch_size
        self._output_dim = patch_size**2 * image_channels

        self._num_layers = [int(i) for i in num_layers.split(",")]
        self._latent_slots = latent_slots
        self._time_on_latent = time_on_latent
        self._cond_on_latent = cond_on_latent_n > 0
        if self._time_on_latent:  # replace 1 latent with time emb.
            latent_slots -= 1
        latent_slots -= cond_on_latent_n
        self._latent_dim = latent_dim
        self._tape_slots = self._num_tokens
        self._tape_dim = tape_dim
        self._cond_dim = cond_dim = cond_dim if cond_dim > 0 else tape_dim
        self._latent_pos_encoding = latent_pos_encoding
        self._tape_pos_encoding = tape_pos_encoding
        assert self_cond in ["none", "latent", "latent+tape", "tape"]
        self._self_cond = self_cond
        self._cond_tape_writable = cond_tape_writable
        self._cond_decoupled_read = cond_decoupled_read
        self.stem_ln = keras.layers.LayerNormalization(epsilon=1e-6)
        self.time_emb = ScalarEmbedding(
            dim=(latent_dim if self._time_on_latent else cond_dim) // 4,
            scaling=time_scaling,
            expansion=4,
        )
        if cond_proj:
            self.cond_proj = keras.layers.Dense(latent_dim if self._cond_on_latent else cond_dim)
        else:
            self.cond_proj = keras.layers.Identity()

        self.make_latent_pos(latent_slots, latent_dim, latent_pos_encoding, time_scaling)
        self.make_tape_pos(tape_dim, tape_pos_encoding, time_scaling)

        if self_cond in ["latent", "latent+tape"]:
            self.latent_prev_proj = MLP(
                num_layers=1,
                dim=latent_dim,
                mlp_ratio=latent_mlp_ratio,
                drop_path=0.0,
                drop_units=0.0,
            )
            self.latent_prev_ln = keras.layers.LayerNormalization(epsilon=1e-6, gamma_initializer="zeros")

        if self_cond in ["tape", "latent+tape"]:
            self.tape_prev_proj = MLP(
                num_layers=1,
                dim=tape_dim,
                mlp_ratio=tape_mlp_ratio,
                drop_path=0.0,
                drop_units=0.0,
            )
            self.tape_prev_ln = keras.layers.LayerNormalization(epsilon=1e-6, gamma_initializer="zeros")

        self.read_units = torch.nn.ModuleList()
        self.read_cond_units = torch.nn.ModuleList()
        self.write_units = torch.nn.ModuleList()
        self.latent_processing_units = torch.nn.ModuleList()

        for num_layers_per_readwrite in self._num_layers:
            self.read_units.append(
                TransformerDecoderLayer(
                    dim=latent_dim,
                    mlp_ratio=latent_mlp_ratio,
                    num_heads=rw_num_heads,
                    drop_path=0.0,
                    drop_units=0.0,
                    drop_att=0.0,
                    dim_x_att=min(tape_dim, latent_dim),
                    self_attention=False,
                    cross_attention=True,
                    use_mlp=True,
                    use_enc_ln=xattn_enc_ln,
                )
            )
            if cond_decoupled_read:
                self.read_cond_units.append(
                    TransformerDecoderLayer(
                        dim=latent_dim,
                        mlp_ratio=latent_mlp_ratio,
                        num_heads=rw_num_heads,
                        drop_path=0.0,
                        drop_units=0.0,
                        drop_att=0.0,
                        dim_x_att=min(cond_dim, latent_dim),
                        self_attention=False,
                        cross_attention=True,
                        use_mlp=True,
                        use_enc_ln=xattn_enc_ln,
                    )
                )
            if num_layers_per_readwrite == 0:
                self.write_units.append(LambdaModule(lambda x: x))
                self.latent_processing_units.append(LambdaModule(lambda x, _: x))
            else:
                self.write_units.append(
                    TransformerDecoderLayer(
                        dim=tape_dim,
                        mlp_ratio=tape_mlp_ratio,
                        num_heads=rw_num_heads,
                        drop_path=0.0,
                        drop_units=0.0,
                        drop_att=0.0,
                        dim_x_att=min(tape_dim, latent_dim),
                        self_attention=False,
                        cross_attention=True,
                        use_mlp=True if tape_mlp_ratio > 0 else False,
                        use_enc_ln=xattn_enc_ln,
                    )
                )
                self.latent_processing_units.append(
                    TransformerEncoder(
                        num_layers=num_layers_per_readwrite,
                        dim=latent_dim,
                        mlp_ratio=latent_mlp_ratio,
                        num_heads=latent_num_heads,
                        drop_path=drop_path,
                        drop_units=drop_units,
                        drop_att=drop_att,
                    )
                )

        self.output_ln = keras.layers.LayerNormalization(epsilon=1e-6, center=True, scale=True)
        self.output_linear = keras.layers.Dense(self._output_dim)

        self.stem = keras.layers.Conv2D(
            filters=tape_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
            use_bias=True,
            data_format="channels_first",  # to get pytorch channel order
        )

    def make_latent_pos(
        self,
        latent_slots: int,
        latent_dim: int,
        latent_pos_encoding: str,
        time_scaling: float,
    ) -> None:
        if latent_pos_encoding in ["sin_cos", "sin_cos_plus_learned"]:
            self.register_buffer(
                "latent_pos_emb",
                create_2d_sin_cos_pos_emb(
                    n_rows=latent_slots,
                    n_cols=1,
                    dim=latent_dim,
                    normalization_max=time_scaling,
                ),
            )
        if latent_pos_encoding == "learned":
            init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
            self.latent_pos_emb = torch.nn.Parameter(init((latent_slots, latent_dim)))
        elif latent_pos_encoding == "sin_cos_plus_learned":
            init = keras.initializers.zeros()
            self.latent_pos_emb_res = torch.nn.Parameter(init(latent_slots, latent_dim))
        else:
            raise ValueError(f"Unknown latent_pos_encoding `{latent_pos_encoding}`")

    def make_tape_pos(
        self,
        tape_dim: int,
        tape_pos_encoding: str,
        time_scaling: float,
    ) -> None:
        if tape_pos_encoding in ["sin_cos", "sin_cos_plus_learned"]:
            self.register_buffer(
                "tape_pos_emb",
                create_2d_sin_cos_pos_emb(
                    n_rows=self._n_rows,
                    n_cols=self._n_cols,
                    dim=tape_dim,
                    normalization_max=time_scaling,
                ),
            )
        if tape_pos_encoding == "learned":
            init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
            self.tape_pos_emb = torch.nn.Parameter(init((self._n_rows * self._n_cols, tape_dim)))
        elif tape_pos_encoding == "sin_cos_plus_learned":
            param = keras.initializers.zeros()(shape=(self._n_rows * self._n_cols, tape_dim))
            self.tape_pos_emb_res = torch.nn.Parameter(param)
        else:
            raise ValueError(f"Unknown tape_pos_encoding `{tape_pos_encoding}`")

    def initialize_cond(
        self,
        t: torch.Tensor | None,
        cond: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if t is not None:
            t = self.time_emb(t, last_swish=False, normalize=True)
            t = rearrange(t, "b d -> b 1 d")
        if cond is not None:
            cond = self.cond_proj(cond)
            if cond.ndim == 2:
                cond = rearrange(cond, "b d -> b 1 d")
        return t, cond

    def initialize_tape(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor | None,
        cond: torch.Tensor | None,
        tape_prev: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tape_r = None
        if not self._time_on_latent and time_emb is not None:
            tape_r = time_emb
        if not self._cond_on_latent and cond is not None:
            tape_r = _concat_tokens(tape_r, cond)

        tape = self.stem(x)
        tape = rearrange(tape, "b d h w -> b (h w) d")
        tape_pos_emb = rearrange(self.tape_pos_emb, "n d -> 1 n d")
        if self._tape_pos_encoding in ["sin_cos_plus_learned"]:
            tape_pos_emb += rearrange(self.tape_pos_emb_res, "n d -> 1 n d")
        tape = self.stem_ln(tape) + tape_pos_emb

        if self._self_cond in ["tape", "latent+tape"] and tape_prev is not None:
            tape = tape + self.tape_prev_ln(self.tape_prev_proj(tape_prev))
        if self._cond_tape_writable and tape_r is not None:
            tape, tape_r = _concat_tokens(tape, tape_r), None

        return tape, tape_r

    def initialize_latent(
        self,
        batch_size: int,
        time_emb: torch.Tensor | None,
        cond: torch.Tensor | None,
        latent_prev: torch.Tensor | None,
    ) -> torch.Tensor:
        latent = self.latent_pos_emb
        if self._latent_pos_encoding in ["sin_cos_plus_learned"]:
            latent = latent + self.latent_pos_emb_res
        latent = latent.repeat(batch_size, 1, 1)
        if self._time_on_latent and time_emb is not None:
            latent = _concat_tokens(latent, time_emb)
        if self._cond_on_latent and cond is not None:
            latent = _concat_tokens(latent, cond)
        if self._self_cond in ["latent", "latent+tape"] and latent_prev is not None:
            latent = latent + self.latent_prev_ln(self.latent_prev_proj(latent_prev))
        return latent

    def compute(
        self,
        latent: torch.Tensor,
        tape: torch.Tensor,
        tape_r: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for i in range(len(self._num_layers)):
            if self._cond_decoupled_read:
                latent = self.read_cond_units[i](latent, tape_r)
                latent = self.read_units[i](latent, tape)
            else:
                tape_merged = _concat_tokens(tape, tape_r)
                latent = self.read_units[i](latent, tape_merged)
            latent = self.latent_processing_units[i](latent)
            tape = self.write_units[i](tape, latent)
        return latent, tape

    def readout_tape(self, tape: torch.Tensor) -> torch.Tensor:
        tokens = self.output_linear(self.output_ln(tape[:, : self._num_tokens]))
        tokens = rearrange(
            tokens,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=self._n_rows,
            w=self._n_cols,
            p1=self._patch_size,
            p2=self._patch_size,
        )
        return tokens

    @property
    def latent_shape(self) -> list[int]:
        return [self._latent_slots, self._latent_dim]

    @property
    def tape_shape(self) -> list[int]:
        return [self._tape_slots, self._tape_dim]

    @property
    def image_shape(self) -> list[int]:
        return [self._image_channels, self._image_height, self._image_width]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def pass_dummy_data(self, num_classes: int | None = None) -> None:
        # pass dummy data to initialize weights
        was_training = self.training
        self.eval()

        self(
            x=torch.zeros([1, *self.image_shape], device=self.device),
            t=0.0,
            cond=None if num_classes is None else torch.zeros([1, num_classes], device=self.device),
        )

        self.train(was_training)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float,
        cond: torch.Tensor | None = None,
        latent_prev: torch.Tensor | None = None,
        tape_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x.ndim == 4
        bs = x.shape[0]

        if isinstance(t, float) or t.ndim == 0:
            t = torch.full((bs,), t, device=x.device, dtype=torch.float32)

        if latent_prev is None:
            latent_prev = torch.zeros(bs, *self.latent_shape, device=x.device)

        if tape_prev is None:
            tape_prev = torch.zeros(bs, *self.tape_shape, device=x.device)

        if self._cond_on_latent and cond is None:
            raise ValueError("cond is None but cond_on_latent is True")

        time_emb, cond = self.initialize_cond(t, cond)
        tape, tape_r = self.initialize_tape(x, time_emb, cond, tape_prev)
        latent = self.initialize_latent(bs, time_emb, cond, latent_prev)
        latent, tape = self.compute(latent, tape, tape_r)
        x = self.readout_tape(tape)
        return x, latent, tape[:, : self._tape_slots]

    def load_weights_numpy(self, np_file):
        # load weights from numpy file relying on the order of parameters
        weights_np = list(np.load(np_file, allow_pickle=True).item().values())
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        for weight_np, param in zip(weights_np, trainable_params):
            data = torch.from_numpy(weight_np).to(param.device)
            param.data.copy_(data)

    def save_weights_numpy(self, np_file):
        # save weights to numpy file relying on the order of parameters
        weights_np = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                weights_np[name] = param.detach().cpu().numpy()

        np.save(np_file, weights_np)
