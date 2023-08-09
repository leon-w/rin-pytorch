import keras_core as keras
import torch
import torch_optimizer

from .lamb import Lamb


def get_optimizer(name: str, params, lr: float, **kwargs) -> torch.optim.Optimizer:
    name = name.lower()

    if name == "lamb":
        return Lamb(params=params, lr=lr, **kwargs)

    optimizer_cls = getattr(torch.optim, name, None)
    if optimizer_cls is None:
        optimizer_cls = torch_optimizer.get(name)

    return optimizer_cls(params=params, lr=lr, **kwargs)


def build_torch_parameters_to_keras_names_mapping(model: torch.nn.Module) -> dict[int, str]:
    mapping = {}

    def map_params(x):
        if isinstance(x, keras.layers.Layer):
            # due to a bug in keras_core (already fixed but not published yet)
            # we need to do some extra work for Dense layers
            if isinstance(x, keras.layers.Dense):
                mapping[id(x.kernel.value)] = "kernel"
                if x.use_bias:
                    mapping[id(x.bias.value)] = "bias"
            else:
                for w in x.trainable_weights:
                    mapping[id(w.value)] = w.name

    model.apply(map_params)

    return mapping


def override_config_for_names(
    parameters,
    names: list[str],
    config_override: dict,
    name_mapping: dict[int, str],
) -> list[dict]:
    group_default = {"params": []}
    group_override = {"params": [], **config_override}

    for param in parameters:
        if not param.requires_grad:
            continue

        if id(param) in name_mapping and any(name in name_mapping[id(param)] for name in names):
            group_override["params"].append(param)
        else:
            group_default["params"].append(param)

    if len(group_default["params"]) == 0:
        return [group_override]
    elif len(group_override["params"]) == 0:
        return [group_default]
    else:
        return [group_default, group_override]
