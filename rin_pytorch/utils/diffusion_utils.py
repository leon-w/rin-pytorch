import math

import torch


class Scheduler:
    """Time scheduling and add noise to data."""

    def __init__(self, train_schedule):
        self._time_transform = self.get_time_transform(train_schedule)

    def get_time_transform(self, schedule_name):
        """Returns time transformation function according to schedule name."""
        if schedule_name.startswith("log@"):
            start, end, reverse = schedule_name.split("@")[1].split(",")
            start, end = float(start), float(end)
            reverse = reverse.lower() in ["t", "true"]
            return lambda t: log_schedule(t, start, end, reverse)
        elif schedule_name.startswith("sigmoid@"):
            start, end, tau = schedule_name.split("@")[1].split(",")
            start, end, tau = float(start), float(end), float(tau)
            return lambda t: sigmoid_schedule(t, start, end, tau)
        elif schedule_name.startswith("cosine"):
            if "@" in schedule_name:
                start, end, tau = schedule_name.split("@")[1].split(",")
                start, end, tau = float(start), float(end), float(tau)
                return lambda t: cosine_schedule(t, start, end, tau)
            else:
                return lambda t: cosine_schedule_simple(t)
        elif schedule_name.startswith("simple_linear"):
            return lambda t: simple_linear_schedule(t)
        else:
            raise ValueError(f"Unknown train schedule `{schedule_name}`")

    def time_transform(self, time_step):
        return self._time_transform(time_step)

    def sample_noise(self, shape, device=None, seed=None):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        return torch.randn(shape, device=device, generator=generator)

    def add_noise(
        self,
        inputs: torch.Tensor,
        t: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = inputs.device
        time_step_shape = [inputs.size(0)] + [1] * (inputs.ndim - 1)
        if t is None:
            t = torch.rand(time_step_shape, device=device)
        elif isinstance(t, float):
            t = torch.full(time_step_shape, t, device=device)
        else:
            t = t.reshape(time_step_shape)

        gamma = self.time_transform(t)
        noise = self.sample_noise(inputs.shape, device=device)
        inputs_noised = inputs * torch.sqrt(gamma) + noise * torch.sqrt(1 - gamma)

        return inputs_noised, noise, t.squeeze(), gamma

    def transition_step(self, samples, data_pred, noise_pred, gamma_now, gamma_prev, sampler_name):
        """Transition to states with a smaller time step."""
        ddpm_var_type = "large"
        if sampler_name.startswith("ddpm") and "@" in sampler_name:
            ddpm_var_type = sampler_name.split("@")[1]

        if sampler_name == "ddim":
            samples = data_pred * torch.sqrt(gamma_prev) + noise_pred * torch.sqrt(1 - gamma_prev)
        elif sampler_name.startswith("ddpm"):
            log_alpha_t = torch.log(gamma_now) - torch.log(gamma_prev)
            alpha_t = torch.clamp(torch.exp(log_alpha_t), 0.0, 1.0)
            x_mean = torch.rsqrt(alpha_t) * (samples - torch.rsqrt(1 - gamma_now) * (1 - alpha_t) * noise_pred)
            if ddpm_var_type == "large":
                var_t = 1.0 - alpha_t  # var = beta_t
            elif ddpm_var_type == "small":
                var_t = torch.exp(torch.log1p(-gamma_prev) - torch.log1p(-gamma_now)) * (1.0 - alpha_t)
            else:
                raise ValueError(f"Unknown ddpm_var_type {ddpm_var_type}")
            eps = self.sample_noise(data_pred.shape, device=data_pred.device)
            samples = x_mean + torch.sqrt(var_t) * eps
        return samples


def float32(x, device=None):
    return torch.tensor(x, dtype=torch.float32, device=device)


def cosine_schedule_simple(t, ns=0.0002, ds=0.00025):
    """Cosine schedule.

    Args:
      t: `float` between 0 and 1.
      ns: `float` numerator constant shift.
      ds: `float` denominator constant shift.

    Returns:
      `float` of transformed time between 0 and 1.
    """
    return torch.cos(((t + ns) / (1 + ds)) * math.pi / 2) ** 2


def cosine_schedule(t, start=0.0, end=0.5, tau=1.0, clip_min=1e-9):
    """Cosine schedule.

    Args:
        t: `float` between 0 and 1.
        start: `float` starting point in x-axis of cosine function.
        end: `float` ending point in x-axis of cosine function.
        tau: `float` temperature.
        clip_min: `float` lower bound for output.

    Returns:
        `float` of transformed time between 0 and 1.
    """
    start = float32(start, device=t.device)
    end = float32(end, device=t.device)

    y_start = torch.cos(start * math.pi / 2) ** (2 * tau)
    y_end = torch.cos(end * math.pi / 2) ** (2 * tau)
    output = (torch.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau) - y_end) / (y_start - y_end)
    output.clamp_(clip_min, 1.0)
    return output


def sigmoid_schedule(t, start=-3.0, end=3.0, tau=1.0, clip_min=1e-9):
    """Sigmoid schedule.

    Args:
        t: `float` between 0 and 1.
        start: `float` starting point in x-axis of sigmoid function.
        end: `float` ending point in x-axis of sigmoid function.
        tau: `float` scaling temperature for sigmoid function.
        clip_min: `float` lower bound for output.

    Returns:
        `float` of transformed time between 0 and 1.
    """
    start = float32(start, device=t.device)
    end = float32(end, device=t.device)

    v_start = torch.sigmoid(start / tau)
    v_end = torch.sigmoid(end / tau)
    output = (-torch.sigmoid((t * (end - start) + start) / tau) + v_end) / (v_end - v_start)
    output.clamp_(clip_min, 1.0)
    return output


def log_schedule(t, start=1.0, end=100.0, reverse=False):
    """Log schedule.

    Args:
      t: `float` between 0 and 1.
      start: `float` starting point in x-axis of log function.
      end: `float` ending point in x-axis of log function.
      reverse: `boolean` whether to reverse the curving direction.

    Returns:
      `float` of transformed time between 0 and 1.
    """
    if reverse:
        start, end = end, start

    start = float32(start, device=t.device)
    end = float32(end, device=t.device)

    v_start = start.log()
    v_end = end.log()
    output = (-torch.log(t * (end - start) + start) + v_end) / (v_end - v_start)
    output.clamp_(0.0, 1.0)
    return output


def simple_linear_schedule(t, clip_min=1e-9):
    """Simple linear schedule.

    Args:
        t: `float` between 0 and 1.
        clip_min: `float` lower bound for output.

    Returns:
        `float` of transformed time between 0 and 1.
    """
    output = 1.0 - t
    output.clamp_(clip_min, 1.0)
    return output


def get_x0_clipping_function(x0_clip):
    """Get x0 clipping function."""
    if x0_clip is None or x0_clip == "":
        return lambda x: x
    else:
        x0_min, x0_max = x0_clip.split(",")
        x0_min, x0_max = float(x0_min), float(x0_max)
        return lambda x: torch.clamp(x, x0_min, x0_max)


def get_x0_from_eps(xt, gamma, noise_pred):
    data_pred = 1.0 / torch.sqrt(gamma) * (xt - torch.sqrt(1.0 - gamma) * noise_pred)
    return data_pred


def get_eps_from_x0(xt, gamma, data_pred):
    noise_pred = 1.0 / torch.sqrt(1 - gamma) * (xt - torch.sqrt(gamma) * data_pred)
    return noise_pred


def get_x0_from_v(xt, gamma, v_pred):
    return torch.sqrt(gamma) * xt - torch.sqrt(1 - gamma) * v_pred


def get_eps_from_v(xt, gamma, v_pred):
    return torch.sqrt(1 - gamma) * xt + torch.sqrt(gamma) * v_pred


def get_x0_eps(
    xt,
    gamma,
    denoise_out,
    pred_type,
    truncate_noise=False,
    clip_x0=True,
):
    """Get x0 and eps from denoising output."""
    if pred_type == "eps":
        noise_pred = denoise_out
        data_pred = get_x0_from_eps(xt, gamma, noise_pred)
        if clip_x0:
            data_pred.clamp_(-1.0, 1.0)
        if truncate_noise:
            noise_pred = get_eps_from_x0(xt, gamma, data_pred)
    elif pred_type.startswith("x"):
        data_pred = denoise_out
        if clip_x0:
            data_pred.clamp_(-1.0, 1.0)
        noise_pred = get_eps_from_x0(xt, gamma, data_pred)
    elif pred_type.startswith("v"):
        v_pred = denoise_out
        data_pred = get_x0_from_v(xt, gamma, v_pred)
        if clip_x0:
            data_pred.clamp_(-1.0, 1.0)
        if truncate_noise:
            noise_pred = get_eps_from_x0(xt, gamma, data_pred)
        else:
            noise_pred = get_eps_from_v(xt, gamma, v_pred)
    else:
        raise ValueError(f"Unknown pred_type `{pred_type}`")

    return {"noise_pred": noise_pred, "data_pred": data_pred}


def get_self_cond_estimate(
    data_pred,
    noise_pred,
    self_cond,
    pred_type,
):
    """Returns self cond estimate given predicted data or noise."""
    assert self_cond in ["x", "eps", "auto"]
    if self_cond == "x":
        estimate = data_pred
    elif self_cond == "eps":
        estimate = noise_pred
    else:
        estimate = noise_pred if pred_type == "eps" else data_pred
    return estimate
