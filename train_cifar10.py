import torchvision

from rin_pytorch import Rin, RinDiffusionModel, Trainer

config = dict(
    rin=dict(
        num_layers="2,2,2",
        latent_slots=128,
        latent_dim=512,
        latent_mlp_ratio=4,
        latent_num_heads=16,
        tape_dim=256,
        tape_mlp_ratio=2,
        rw_num_heads=8,
        image_height=32,
        image_width=32,
        image_channels=3,
        patch_size=2,
        latent_pos_encoding="learned",
        tape_pos_encoding="learned",
        drop_path=0.1,
        drop_units=0.1,
        drop_att=0.0,
        time_scaling=1000,
        self_cond="latent",
        time_on_latent=True,
        cond_on_latent_n=1,
        cond_tape_writable=False,
        cond_dim=0,
        cond_proj=True,
        cond_decoupled_read=False,
        xattn_enc_ln=False,
    ),
    diffusion=dict(
        train_schedule="sigmoid@-3,3,0.9",
        inference_schedule="cosine",
        pred_type="eps",
        self_cond="latent",
        loss_type="eps",
    ),
    trainer=dict(
        num_classes=10,
        train_num_steps=150_000,
        train_batch_size=256,
        split_batches=True,
        fp16=False,
        amp=False,
        lr_scheduler_name="cosine",
        lr=3e-3,
        lr_warmup_steps=10_000,
        optimizer_name="lamb",
        optimizer_exclude_weight_decay=["bias", "beta", "gamma"],
        optimizer_kwargs=dict(weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8),
        clip_grad_norm=None,
        sample_every=1000,
        num_dl_workers=4,
        ema_decay=0.9999,
        ema_update_every=1,
        sampling_kwargs=dict(iterations=100, method="ddim"),
        checkpoint_folder="results/cifar10",
        run_name="rin_cifar10",
        log_to_wandb=True,
    ),
)


rin = Rin(**config["rin"]).cuda()
rin.pass_dummy_data(num_classes=10)  # populate lazy model with weights
diffusion_model = RinDiffusionModel(rin=rin, **config["diffusion"])

rin_ema = Rin(**config["rin"]).cuda()
rin_ema.pass_dummy_data(num_classes=10)
ema_diffusion_model = RinDiffusionModel(rin=rin_ema, **config["diffusion"])


dataset = torchvision.datasets.CIFAR10(
    "datasets/cifar10",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomHorizontalFlip(),
        ]
    ),
)


trainer = Trainer(
    diffusion_model,
    ema_diffusion_model,
    dataset,
    **config["trainer"],
)

trainer.train()
