import torch
from torch.optim.optimizer import Optimizer


# implementation based on https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/lamb.py
# adapted to work like https://github.com/tensorflow/addons/blob/v0.20.0/tensorflow_addons/optimizers/lamb.py
class Lamb(Optimizer):
    r"""Implements Lamb algorithm.

    It has been proposed in `Large Batch Optimization for Deep Learning:
    Training BERT in 76 minutes`__.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        disable_layer_adaption: always use trust ratio = 1, which turns this
            into Adam. Useful for comparison purposes. (default: False)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.Lamb(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/1904.00962

    Note:
        Reference code: https://github.com/cybertronai/pytorch-lamb
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0,
        disable_layer_adaption: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            disable_layer_adaption=disable_layer_adaption,
        )

        super().__init__(params, defaults)

    def step(self, closure=None):
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = "Lamb does not support sparse gradients, " "please consider SparseAdam instead"
                    raise RuntimeError(msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m, v = state["m"], state["v"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                m_t = m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v_t = v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_t_hat = m_t / (1.0 - beta1 ** state["step"])
                v_t_hat = v_t / (1.0 - beta2 ** state["step"])

                update = m_t_hat.div_(v_t_hat.sqrt_().add_(group["eps"]))

                if group["weight_decay"] != 0:
                    update.add_(p.data, alpha=group["weight_decay"])

                w_norm = torch.norm(p.data)
                g_norm = torch.norm(update)

                if w_norm == 0 or g_norm == 0 or group["disable_layer_adaption"]:
                    trust_ratio = 1
                else:
                    trust_ratio = w_norm / g_norm

                p.data.add_(update, alpha=-group["lr"] * trust_ratio)

        return loss
