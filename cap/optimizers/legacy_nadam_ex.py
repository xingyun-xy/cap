from typing import Any, Callable, Dict, List

import changan_plugin_pytorch as changan
import torch
from torch.optim.optimizer import Optimizer

from cap.registry import OBJECT_REGISTRY

__all__ = [
    "LegacyNadamEx",
]


@OBJECT_REGISTRY.register
class LegacyNadamEx(Optimizer):
    """Nadam optimizer.

    This optimizer compute wd in the wrong way, but get far more better
     performance.

    Args:
        params: Parameters.
        rescale_grad: Coefficient of rescale grad.
        lr: Learning rate.
        weight_decay (float, optional): _description_. Defaults to 5e-5.
        beta1: Exponential decay rate for the first moment estimates.
        beta2: Exponential decay rate for the second moment estimates.
        epsilon: Epsilon, small value to avoid division by 0.
        schedule_decay: Exponential decay rate for the momentum schedule.
        wd_type: The way to compute weight decay, possible values are
                        {1, 2, 3}. Defaults to 3.
        fused: Whether to use changan `multi_tensor_legacynadamex`,
                which can deal with multi param, and more faster.
    """

    def __init__(
        self,
        params: Dict,
        rescale_grad: float = 1.0,
        lr: float = 0.001,
        weight_decay: float = 5e-5,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        schedule_decay: float = 0.004,
        wd_type: int = 3,
        fused: bool = False,
    ):

        if not (wd_type == 3):
            raise NotImplementedError  # not implemented

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= epsilon:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta1 parameter {}".format(beta1))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta2 parameter: {}".format(beta2))
        if not 0.0 <= schedule_decay:
            raise ValueError(
                "Invalid shedule_decay value: {}".format(schedule_decay)
            )  # noqa E501

        self.fused = fused

        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "weight_decay": weight_decay,
            "schedule_decay": schedule_decay,
            "rescale_grad": rescale_grad,
            "m_schedule": 1.0,
        }

        super(LegacyNadamEx, self).__init__(params, defaults)

    def __setstate__(self, state: Dict) -> None:
        super(LegacyNadamEx, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("schedule_decay", 0.004)
            group.setdefault("weight_decay", 5e-5)

    @torch.no_grad()
    def step(self, closure: Callable = None) -> Any:
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            mean = []
            variance = []
            state_steps = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                grads.append(p.grad)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["mean"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["variance"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                mean.append(state["mean"])
                variance.append(state["variance"])
                state["step"] += 1
                state_steps.append(state["step"])

            if self.fused:
                group["m_schedule"] = changan.multi_tensor_legacynadamex(
                    [params_with_grad, grads, mean, variance],
                    state_steps,
                    group["lr"],
                    group["weight_decay"],
                    group["beta1"],
                    group["beta2"],
                    group["epsilon"],
                    group["schedule_decay"],
                    group["m_schedule"],
                    group["rescale_grad"],
                )
            else:
                group["m_schedule"] = self.legacynadamex(
                    params=params_with_grad,
                    grads=grads,
                    mean=mean,
                    variance=variance,
                    state_steps=state_steps,
                    lr=group["lr"],
                    wd=group["weight_decay"],
                    beta1=group["beta1"],
                    beta2=group["beta2"],
                    eps=group["epsilon"],
                    schedule_decay=group["schedule_decay"],
                    m_schedule=group["m_schedule"],
                    rescale_grad=group["rescale_grad"],
                )

        return loss

    @staticmethod
    def legacynadamex(
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        mean: List[torch.Tensor],
        variance: List[torch.Tensor],
        state_steps: List[int],
        lr: float,
        wd: float,
        beta1: float,
        beta2: float,
        eps: float,
        schedule_decay: float,
        m_schedule: float,
        rescale_grad: float,
    ) -> float:
        for i, param in enumerate(params):
            grad = grads[i]
            mean_i = mean[i]
            variance_i = variance[i]
            t = state_steps[i]

            # preprocess grad
            if wd != 0:
                if grad.is_sparse:
                    raise RuntimeError(
                        "weight_decay option is not \
                         compatible with sparse gradients"
                    )
                grad.mul_(torch.add(param.mul(wd), rescale_grad))

            else:
                grad.mul_(rescale_grad)

            # warming momentum schedule
            momentum_t = beta1 * (1.0 - 0.5 * (pow(0.96, t * schedule_decay)))
            momentum_t_1 = beta1 * (
                1.0 - 0.5 * (pow(0.96, (t + 1) * schedule_decay))
            )

            m_schedule = m_schedule * momentum_t
            m_schedule_next = m_schedule * momentum_t_1

            # update m_t and v_t
            m_t, v_t = mean_i, variance_i

            m_t.mul_(beta1)
            m_t.add_(grad, alpha=1.0 - beta1)
            v_t.mul_(beta2)
            v_t.add_(grad ** 2, alpha=1.0 - beta2)

            grad_prime = torch.div(grad, 1.0 - m_schedule)
            m_t_prime = torch.div(m_t, 1.0 - m_schedule_next)
            v_t_prime = torch.div(v_t, 1.0 - pow(beta2, t))

            grad_prime.mul_(1.0 - momentum_t)
            m_t_prime.mul_(momentum_t_1)
            m_t_bar = torch.add(grad_prime, m_t_prime)

            # update weight
            m_t_bar.mul_(lr)
            m_t_bar.div_(v_t_prime.sqrt_().add_(eps))
            param.sub_(m_t_bar)

        return m_schedule
