
# AdaptiveScheduler 📈🌀

**AdaptiveScheduler** is a general-purpose, gradient-agnostic learning rate & momentum scheduler.
It adapts hyperparameters online using **loss signals**, **reward signals**, and **variance trends**, while remaining optimizer-agnostic.

---

## 🔹 Why AdaptiveScheduler?

* **Optimizers** should only care about *how to apply gradients*.
* **Schedulers** should decide *how hyperparameters evolve*.

Traditional schedulers (StepLR, CosineAnnealingLR, etc.) only depend on **time**.
**AdaptiveScheduler** extends this by also responding to **feedback**:

* Loss trend (better/worse).
* Loss variance (stable/noisy).
* Optional reward signals (RL, bandit, meta-learning).
* Patience & cooldown (don’t thrash).
* Optional cosine restarts (exploration).

---

## 🔹 Core Theory

### Inputs

* Scalar **loss** (required).
* Scalar **reward** (optional).
* Moving statistics (variance, EMA).

### State (per parameter group)

* Current learning rate (LR).
* Current momentum (if present).
* History: EMA loss, EMA reward, loss variance.
* Counters: steps since improvement, cooldowns, restart timers.

### Policies

* **Trend-aware**

  $$
  \text{if } \Delta \ell < 0: \ \alpha \leftarrow \alpha \cdot u \quad 
  \text{else } \alpha \leftarrow \alpha \cdot d
  $$

  *(u > 1, d < 1)*

* **Variance-aware**

  $$
  \alpha \leftarrow \frac{\alpha}{1 + \lambda \cdot \mathrm{Var}(\ell)}
  $$

* **Reward-aware**

  $$
  R_t > R_{t-1} \ \Rightarrow \ \alpha \uparrow, \ \mu \uparrow \quad 
  R_t < R_{t-1} \ \Rightarrow \ \alpha \downarrow, \ \mu \downarrow
  $$

* **Patience & Cooldown**

  * No improvement for *p* steps → big LR drop.
  * Wait *c* steps before dropping again.

* **Cosine Restarts**

  * Smooth periodic resets for exploration.

### Outputs

* Updates `optimizer.param_groups[i]["lr"]` and `["momentum"]` (if present).
* Never inspects weights or gradients.

---

## 🔹 Installation

Clone this repo and import directly:

```bash
git clone https://github.com/yourname/AdaptiveScheduler.git
cd AdaptiveScheduler
```

---

## 🔹 Usage

```python
import torch
from adaptive_scheduler import AdaptiveScheduler

model = torch.nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

scheduler = AdaptiveScheduler(
    optimizer,
    up_factor=1.03, down_factor=0.8,
    patience=30, cooldown=60,
    use_cosine=True, T0=200, T_mult=2
)

for step, batch in enumerate(loader):
    loss = compute_loss(batch)
    loss.backward()
    optimizer.step()
    scheduler.step_loss(loss)   # update from scalar loss only
    optimizer.zero_grad()
```

---

## 🔹 Features

* ✅ **Gradient-free** (no param/grad access).
* ✅ **Optimizer-agnostic** (works with SGD, AdamW, Adafactor, etc.).
* ✅ **Mix-and-match rules** (trend, variance, reward).
* ✅ **Patience & cooldown** to prevent thrashing.
* ✅ **Safe bounds** for LR and momentum.
* ✅ **Optional cosine warm restarts**.
* ✅ **Serializable** via `state_dict()`.

---

## 🔹 Example Configurations

* **Conservative training**

  ```python
  AdaptiveScheduler(optimizer, up_factor=1.02, down_factor=0.9, patience=40)
  ```

* **Exploratory with cosine**

  ```python
  AdaptiveScheduler(optimizer, use_cosine=True, T0=100, T_mult=2)
  ```

* **Reward-driven RL fine-tuning**

  ```python
  AdaptiveScheduler(optimizer, use_reward=True, reward_up=1.05, reward_down=0.8)
  ```

---

## 🔹 Roadmap

* [ ] TensorBoard logging hooks (LR, momentum vs. loss).
* [ ] JAX/Optax port.
* [ ] Keras `LearningRateSchedule` version.
* [ ] Configurable via YAML/JSON.

---

## 🔹 License

MIT License © 2025 James Theory

### PSEUDOCODE

```python
#!/usr/bin/env python3
"""
Demo: AdaptiveScheduler in action
---------------------------------
Trains a toy regression model on synthetic data.
Shows how the scheduler adapts the learning rate
from scalar loss feedback only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import math
from collections import deque
from typing import Optional, Dict, Any
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class AdaptiveScheduler(_LRScheduler):
    """
    Gradient-agnostic adaptive scheduler.
    Adjusts LR (and momentum if present) from scalar feedback only.
    Never inspects parameters, gradients, or the model.

    Call pattern:
      loss = compute_loss_value_only(...)  # scalar tensor or float
      optimizer.step()
      scheduler.step_loss(loss, reward=maybe_R)
      optimizer.zero_grad()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        # smoothing
        loss_beta: float = 0.9,            # EMA for loss
        reward_beta: float = 0.9,          # EMA for reward
        var_window: int = 20,              # window for running variance
        # trend policy
        up_factor: float = 1.05,           # LR x when improving
        down_factor: float = 0.7,          # LR x when worsening
        min_rel_improve: float = 1e-4,     # improvement threshold
        # variance policy
        var_damp_base: float = 1.0,        # LR /= (base + scale * var)
        var_damp_scale: float = 1.0,
        # reward policy (optional)
        use_reward: bool = False,
        reward_up: float = 1.03,           # LR x when reward trend ↑
        reward_down: float = 0.90,         # LR x when reward trend ↓
        momentum_up: float = 0.01,         # +Δ momentum when reward trend ↑
        momentum_down: float = 0.90,       # × momentum when reward trend ↓
        # patience / cooldown
        patience: int = 20,                # non-improve steps before big drop
        big_drop: float = 0.5,             # LR x on big drop
        cooldown: int = 50,                # steps to wait after big drop
        # cosine restarts overlay (optional)
        use_cosine: bool = False,
        T0: int = 200,
        T_mult: int = 2,
        eta_min_scale: float = 0.1,        # floor = eta_min_scale * current LR
        # bounds
        lr_min: float = 1e-7,
        lr_max: float = 10.0,
        momentum_min: float = 0.0,
        momentum_max: float = 0.999,
        # plumbing
        last_epoch: int = -1,
    ):
      
        self.loss_beta = loss_beta
        self.reward_beta = reward_beta
        self.var_window = var_window

        self.up_factor = up_factor
        self.down_factor = down_factor
        self.min_rel_improve = min_rel_improve

        self.var_damp_base = var_damp_base
        self.var_damp_scale = var_damp_scale

        self.use_reward = use_reward
        self.reward_up = reward_up
        self.reward_down = reward_down
        self.momentum_up = momentum_up
        self.momentum_down = momentum_down

        self.patience = patience
        self.big_drop = big_drop
        self.cooldown = cooldown

        self.use_cosine = use_cosine
        self.T0 = T0
        self.T_mult = T_mult
        self.eta_min_scale = eta_min_scale

        self.lr_min = lr_min
        self.lr_max = lr_max
        self.momentum_min = momentum_min
        self.momentum_max = momentum_max

        # per-param-group state
        self._gstate = []
        for g in optimizer.param_groups:
            base_lr = float(g["lr"])
            base_m = float(g.get("momentum", 0.0))
            self._gstate.append(dict(
                lr=base_lr, base_lr=base_lr,
                m=base_m,  base_m=base_m,
                loss_ema=None, prev_loss=None,
                loss_q=deque(maxlen=self.var_window),
                reward_ema=None, prev_reward=None,
                steps_since_improve=0, cooldown=0,
                T_cur=0, T_i=self.T0,
            ))

        self._pending: Optional[Dict[str, float]] = None
        super().__init__(optimizer, last_epoch)

    # ---- public API (no gradients involved) ----
    def step_loss(self, loss, reward: Optional[float] = None):
        """Consume scalar signals and step the scheduler."""
        # accept torch scalar or float
        lv = float(loss.detach().item() if isinstance(loss, torch.Tensor) else loss)
        rv = None if reward is None else float(reward)
        self._pending = {"loss": lv, "reward": rv}
        return self.step()

    def set_metrics(self, loss, reward: Optional[float] = None):
        """Alternative: set metrics, then call scheduler.step()."""
        self._pending = {
            "loss": float(loss.detach().item() if isinstance(loss, torch.Tensor) else loss),
            "reward": None if reward is None else float(reward),
        }

    # Required by _LRScheduler
    def get_lr(self):
        return [st["lr"] for st in self._gstate]

    def step(self):
        self.last_epoch += 1
        metrics = self._pending
        self._pending = None

        # No metrics? Maintain current LR (still advance cosine if enabled).
        loss = metrics["loss"] if metrics else None
        reward = metrics["reward"] if metrics else None

        for g, st in zip(self.optimizer.param_groups, self._gstate):
            # 1) update EMAs / variance
            if loss is not None:
                st["loss_ema"] = loss if st["loss_ema"] is None else (
                    self.loss_beta * st["loss_ema"] + (1 - self.loss_beta) * loss
                )
                st["loss_q"].append(loss)
            if self.use_reward and (reward is not None):
                st["reward_ema"] = reward if st["reward_ema"] is None else (
                    self.reward_beta * st["reward_ema"] + (1 - self.reward_beta) * reward
                )

            # helpers
            improving = None
            if st["prev_loss"] is not None and st["loss_ema"] is not None:
                prev, cur = st["prev_loss"], st["loss_ema"]
                rel_improve = (prev - cur) / (abs(prev) + 1e-12)
                improving = rel_improve > self.min_rel_improve

            variance = 0.0
            if len(st["loss_q"]) > 1:
                mean = sum(st["loss_q"]) / len(st["loss_q"])
                variance = sum((x - mean) ** 2 for x in st["loss_q"]) / (len(st["loss_q"]) - 1)

            # 2) trend-based LR nudge
            if improving is True:
                st["lr"] *= self.up_factor
                st["steps_since_improve"] = 0
            elif improving is False:
                st["lr"] *= self.down_factor
                st["steps_since_improve"] += 1

            # 3) variance damping
            st["lr"] = st["lr"] / (self.var_damp_base + self.var_damp_scale * variance)

            # 4) reward shaping (optional)
            if self.use_reward and st["reward_ema"] is not None:
                if st["prev_reward"] is not None:
                    if st["reward_ema"] > st["prev_reward"]:
                        st["lr"] *= self.reward_up
                        if "momentum" in g:
                            st["m"] = min(self.momentum_max, st["m"] + self.momentum_up)
                    else:
                        st["lr"] *= self.reward_down
                        if "momentum" in g:
                            st["m"] = max(self.momentum_min, st["m"] * self.momentum_down)
                st["prev_reward"] = st["reward_ema"]

            # 5) patience + cooldown
            if st["cooldown"] > 0:
                st["cooldown"] -= 1
            elif st["steps_since_improve"] >= self.patience:
                st["lr"] *= self.big_drop
                st["steps_since_improve"] = 0
                st["cooldown"] = self.cooldown

            # 6) cosine warm restarts overlay (optional; time-based only)
            if self.use_cosine:
                eta_max = st["lr"]
                eta_min = max(self.lr_min, self.eta_min_scale * eta_max)
                T_i = max(1, st["T_i"])
                cos = 0.5 * (1 + math.cos(math.pi * (st["T_cur"] / T_i)))
                st["lr"] = eta_min + (eta_max - eta_min) * cos

                st["T_cur"] += 1
                if st["T_cur"] >= T_i:
                    st["T_cur"] = 0
                    st["T_i"] = int(T_i * self.T_mult)

            # 7) safety clamps + write back
            st["lr"] = float(min(max(st["lr"], self.lr_min), self.lr_max))
            g["lr"] = st["lr"]
            if "momentum" in g:
                st["m"] = float(min(max(st.get("m", g["momentum"]), self.momentum_min), self.momentum_max))
                g["momentum"] = st["m"]

            if st["loss_ema"] is not None:
                st["prev_loss"] = st["loss_ema"]

        return [st["lr"] for st in self._gstate]

    # serialization
    def state_dict(self) -> Dict[str, Any]:
        base = super().state_dict()
        base["_gstate"] = self._gstate
        return base

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._gstate = state_dict.pop("_gstate")
        super().load_state_dict(state_dict)



def make_toy_dataset(n=512):
    torch.manual_seed(0)
    X = torch.randn(n, 10)
    true_w = torch.randn(10, 1)
    y = X @ true_w + 0.1 * torch.randn(n, 1)
    return TensorDataset(X, y)


def main():
    # Data
    ds = make_toy_dataset()
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    # Model
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    # Adaptive scheduler
    scheduler = AdaptiveScheduler(
        optimizer,
        up_factor=1.03, down_factor=0.8,
        patience=20, cooldown=40,
        use_cosine=True, T0=100, T_mult=2,
        lr_min=1e-5, lr_max=1.0,
    )

    # Training loop
    for epoch in range(5):
        for step, (X, y) in enumerate(loader):
            optimizer.zero_grad()
            pred = model(X)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()

            # Scheduler update from scalar loss
            scheduler.step_loss(loss)

            if step % 20 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch:02d} Step {step:03d} | "
                      f"Loss={loss.item():.4f} | LR={lr:.6f}")


if __name__ == "__main__":
    main()
```
