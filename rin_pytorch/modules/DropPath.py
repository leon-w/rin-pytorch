import torch


class DropPath(torch.nn.Module):
    def __init__(self, drop_rate=0.0):
        super().__init__()

        assert 0.0 <= drop_rate <= 1.0
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_rate == 0.0 or not self.training:
            return x

        keep_rate = 1.0 - self.drop_rate
        drop_mask_shape = [x.shape[0]] + [1] * (x.ndim - 1)
        drop_mask = keep_rate + torch.rand(drop_mask_shape, dtype=x.dtype, device=x.device)
        drop_mask = torch.floor(drop_mask) / keep_rate

        return x * drop_mask
