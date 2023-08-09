import torch
from tqdm import tqdm

from .Rin import Rin
from .utils import diffusion_utils


class RinDiffusionModel(torch.nn.Module):
    def __init__(
        self,
        rin: Rin,
        train_schedule: str,
        inference_schedule: str,
        pred_type: str,
        self_cond: str = "none",
        num_classes: int = 10,
        conditional: str = "class",
        self_cond_rate: float = 0.9,
        loss_type: str = "x",
    ):
        super().__init__()
        self._inference_schedule = inference_schedule
        self._pred_type = pred_type
        self._self_cond = self_cond
        self._num_classes = num_classes
        self._conditional = conditional
        self._self_cond_rate = self_cond_rate
        self._loss_type = loss_type

        self.scheduler = diffusion_utils.Scheduler(train_schedule)

        self.denoiser = rin

    def denoise(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        cond: torch.Tensor | None,
        latent_prev: torch.Tensor | None = None,
        tape_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma = gamma.squeeze()
        assert gamma.ndim == 1
        output, latent, tape = self.denoiser(x, gamma, cond, latent_prev, tape_prev)
        return output, latent, tape

    @torch.no_grad()
    def sample(self, num_samples=64, iterations=100, method="ddim", seed=None, class_override=None):
        samples_shape = [num_samples, *self.denoiser.image_shape]
        device = self.denoiser.device
        if self._conditional == "class":
            # generate random classes
            if class_override is not None:
                cond = torch.full([num_samples], class_override, device=device, dtype=torch.long)
            else:
                generator = None
                if seed is not None:
                    generator = torch.Generator(device=device).manual_seed(seed)
                cond = torch.randint(self._num_classes, [num_samples], device=device, generator=generator)
            cond = torch.nn.functional.one_hot(cond, self._num_classes).float()
        else:
            cond = None

        get_step = lambda t: torch.full([num_samples, 1, 1, 1], 1.0 - t / iterations, device=device)
        if self._inference_schedule is None:
            time_transform = self.scheduler.time_transform
        else:
            time_transform = self.scheduler.get_time_transform(self._inference_schedule)

        samples = self.scheduler.sample_noise(samples_shape, device=device, seed=seed)
        data_pred = torch.zeros_like(samples, device=device)

        latent_prev = None
        tape_prev = None
        for t in tqdm(
            torch.arange(iterations, dtype=torch.float32, device=device), desc="sampling", leave=False, position=1
        ):
            time_step = get_step(t)
            time_step_p = torch.max(get_step(t + 1), torch.tensor(0.0))
            gamma, gamma_prev = time_transform(time_step), time_transform(time_step_p)

            pred_out, latent_prev, tape_prev = self.denoise(samples, gamma, cond, latent_prev, tape_prev)
            x0_eps = diffusion_utils.get_x0_eps(
                samples, gamma, pred_out, self._pred_type, truncate_noise=True, clip_x0=True
            )
            noise_pred, data_pred = x0_eps["noise_pred"], x0_eps["data_pred"]
            samples = self.scheduler.transition_step(
                samples=samples,
                data_pred=data_pred,
                noise_pred=noise_pred,
                gamma_now=gamma,
                gamma_prev=gamma_prev,
                sampler_name=method,
            )

        samples = data_pred * 0.5 + 0.5  # convert -1,1 -> 0,1
        samples.clamp_(0.0, 1.0)

        return samples

    def noise_denoise(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        t: torch.Tensor | None = None,
    ):
        images = images * 2.0 - 1.0
        images_noised, noise, _, gamma = self.scheduler.add_noise(images, t=t)

        bsz = images.size(0)
        latent_prev = torch.zeros((bsz, *self.denoiser.latent_shape), device=images.device)
        tape_prev = torch.zeros((bsz, *self.denoiser.tape_shape), device=images.device)
        if self._self_cond != "none" and self._self_cond_rate > 0.0:
            mask = torch.rand(bsz) < self._self_cond_rate

            if torch.any(mask):
                with torch.no_grad():
                    _, latent_prev_out, tape_prev_out = self.denoise(
                        x=images_noised[mask],
                        gamma=gamma[mask],
                        cond=labels[mask],
                    )

                latent_prev[mask] = latent_prev_out.detach()
                tape_prev[mask] = tape_prev_out.detach()

        denoise_out, _, _ = self.denoise(images_noised, gamma, labels, latent_prev, tape_prev)

        pred_dict = diffusion_utils.get_x0_eps(
            images_noised, gamma, denoise_out, self._pred_type, truncate_noise=False, clip_x0=True
        )
        return images, noise, images_noised, pred_dict

    def compute_loss(
        self,
        images: torch.Tensor,
        noise: torch.Tensor,
        pred_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self._loss_type == "x":
            loss = torch.nn.functional.mse_loss(images, pred_dict["data_pred"])
        elif self._loss_type == "eps":
            loss = torch.nn.functional.mse_loss(noise, pred_dict["noise_pred"])
        else:
            raise ValueError(f"Unknown loss_type `{self._pred_type}`")
        return loss

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        images, noise, _, pred_dict = self.noise_denoise(images, labels, t=t)
        loss = self.compute_loss(images, noise, pred_dict)
        return loss
