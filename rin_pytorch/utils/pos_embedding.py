import torch
from einops import rearrange


def get_angles(
    pos: torch.Tensor,
    i: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    angle_rates = 1 / torch.pow(10000.0, 2 * (i // 2) / dim)
    return pos.float() * angle_rates.float()


def positional_encoding(
    coords: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    angle_rads = get_angles(
        rearrange(coords, "b -> b 1"),
        rearrange(torch.arange(dim, device=coords.device), "d -> 1 1 d"),
        dim,
    )

    # apply sin to even indices in the array; 2i
    angle_rads1 = torch.sin(angle_rads[..., 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads2 = torch.cos(angle_rads[..., 1::2])

    pos_encoding = torch.cat([angle_rads1, angle_rads2], -1)

    return pos_encoding.float()


def get_1d_position_codes(
    seqlen: int,
    out_dim: int,
    normalization_max=6.2831852,
) -> torch.Tensor:
    coords = torch.arange(seqlen, dtype=torch.float32)
    if normalization_max is not None:
        coords = coords / (seqlen - 1) * normalization_max
    coords = positional_encoding(coords, out_dim)
    return coords


def get_2d_position_codes(
    height: int,
    width: int,
    out_dim: int,
    normalization_max=6.2831852,
) -> torch.Tensor:
    y_coords = get_1d_position_codes(height, out_dim // 2, normalization_max)
    y_coords = y_coords.unsqueeze(2)
    y_coords = torch.cat([y_coords, torch.zeros_like(y_coords)], -1)

    x_coords = get_1d_position_codes(width, out_dim // 2, normalization_max)
    x_coords = x_coords.unsqueeze(1)
    x_coords = torch.cat([torch.zeros_like(x_coords), x_coords], -1)

    return y_coords + x_coords


def create_2d_sin_cos_pos_emb(
    n_rows: int,
    n_cols: int,
    dim: int,
    normalization_max=6.2831852,
) -> torch.Tensor:
    if n_rows == 1 or n_cols == 1:
        sin_cos = get_1d_position_codes(n_rows * n_cols, dim, normalization_max=normalization_max)
    else:
        sin_cos = get_2d_position_codes(n_rows, n_cols, dim, normalization_max=normalization_max)
    vis_pos_emb = sin_cos.view(n_rows * n_cols, dim)

    return vis_pos_emb
