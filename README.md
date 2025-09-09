
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

