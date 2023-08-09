from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from diffusers.optimization import get_scheduler as get_lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from tqdm import tqdm

from .RinDiffusionModel import RinDiffusionModel
from .utils.optimization_utils import (
    build_torch_parameters_to_keras_names_mapping,
    get_optimizer,
    override_config_for_names,
)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class Trainer:
    def __init__(
        self,
        diffusion_model: RinDiffusionModel,
        ema_diffusion_model: RinDiffusionModel,  # since RinDiffusionModel can't be copied, we need a second one for EMA
        dataset: Dataset,
        num_classes: int,
        train_num_steps: int,
        train_batch_size=256,
        split_batches=True,
        fp16=False,
        amp=False,
        lr_scheduler_name="cosine",
        lr=1e-4,
        lr_warmup_steps=1000,
        optimizer_name="lamb",
        optimizer_exclude_weight_decay=["bias", "beta", "gamma"],
        optimizer_kwargs=dict(weight_decay=1e-2),
        clip_grad_norm=None,
        sample_every=1000,
        num_dl_workers=2,
        ema_decay=0.9999,
        ema_update_every=1,
        sampling_kwargs=dict(iterations=100, method="ddim"),
        checkpoint_folder="results",
        run_name="rin",
        log_to_wandb=True,
    ):
        self.accelerator = Accelerator(split_batches=split_batches, mixed_precision="fp16" if fp16 else "no")
        self.accelerator.native_amp = amp

        self.diffusion_model = diffusion_model

        self.num_classes = num_classes
        self.train_num_steps = train_num_steps
        self.clip_grad_norm = clip_grad_norm
        self.sample_every = sample_every
        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every
        self.sampling_kwargs = sampling_kwargs

        dl = DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_dl_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        self.optimizer = get_optimizer(
            optimizer_name,
            override_config_for_names(
                self.diffusion_model.parameters(),
                optimizer_exclude_weight_decay,
                {"weight_decay": 0.0, "disable_layer_adaption": True},
                build_torch_parameters_to_keras_names_mapping(self.diffusion_model),
            ),
            lr=lr,
            **optimizer_kwargs,
        )

        self.lr_scheduler = get_lr_scheduler(
            lr_scheduler_name,
            self.optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=train_num_steps,
        )

        self.checkpoint_folder = Path(checkpoint_folder)

        if self.accelerator.is_main_process:
            self.ema_diffusion_model = ema_diffusion_model
            self.ema_diffusion_model.requires_grad_(False)

            self.checkpoint_folder.mkdir(exist_ok=True, parents=True)

        self.step = 0

        self.diffusion_model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.diffusion_model, self.optimizer, self.lr_scheduler
        )

        if self.accelerator.is_main_process:
            wandb.init(project="rin", name=run_name, mode="online" if log_to_wandb else "disabled")

    def save(self, milestone, absolute=False):
        if not self.accelerator.is_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.diffusion_model),
            "ema_model": self.ema_diffusion_model.state_dict(),
            "opt": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }

        if absolute:
            checkpoint_file = milestone
        else:
            checkpoint_file = self.checkpoint_folder / f"model-{milestone}.pt"

        torch.save(data, checkpoint_file)

    def load(self, milestone, absolute=False):
        if absolute:
            checkpoint_file = milestone
        else:
            checkpoint_file = self.checkpoint_folder / f"model-{milestone}.pt"

        data = torch.load(checkpoint_file)

        self.step = data["step"]

        diffusion_model = self.accelerator.unwrap_model(self.diffusion_model)
        diffusion_model.load_state_dict(data["model"])

        if self.accelerator.is_main_process:
            self.ema_diffusion_model.load_state_dict(data["ema_model"])

        self.optimizer.load_state_dict(data["opt"])

        self.lr_scheduler.load_state_dict(data["lr_scheduler"])

    def train(self):
        self.diffusion_model.train()

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not self.accelerator.is_main_process,
            desc="Training",
        ) as pbar:
            while self.step < self.train_num_steps:
                batch_img, batch_class = next(self.dl)
                batch_class = torch.nn.functional.one_hot(batch_class, num_classes=self.num_classes).float()

                self.optimizer.zero_grad()

                loss = self.diffusion_model(batch_img, batch_class)

                self.accelerator.backward(loss)

                if self.clip_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(self.diffusion_model.parameters(), self.clip_grad_norm)

                self.optimizer.step()

                logs = {
                    "loss": loss.item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }

                pbar.set_postfix(logs)
                if self.accelerator.is_main_process:
                    wandb.log(logs, step=self.step)

                self.step += 1
                self.lr_scheduler.step()
                pbar.update(1)

                if self.accelerator.is_main_process:
                    if self.step % self.ema_update_every == 0:
                        # perform ema update
                        with torch.no_grad():
                            for ema_param, param in zip(
                                self.ema_diffusion_model.parameters(),
                                self.diffusion_model.parameters(),
                            ):
                                if param.requires_grad:
                                    ema_param.data.lerp_(param.data, 1 - self.ema_decay)

                    if self.step % self.sample_every == 0:
                        self.ema_diffusion_model.eval()
                        n = 8
                        samples = self.ema_diffusion_model.sample(num_samples=n * n, **self.sampling_kwargs)

                        samples = make_grid(samples, nrow=n, normalize=True, range=(0, 1), padding=0)
                        wandb.log({"samples": [wandb.Image(samples)]}, step=self.step)

                        self.save("latest")

                        del samples
