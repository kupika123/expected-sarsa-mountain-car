# Expected SARSA on Mountain Car — Tabular RL in a Continuous State Space

A from-scratch Expected SARSA agent for Gymnasium's `MountainCar-v0`. The environment has a
continuous two-dimensional state, so the interesting part is not the update rule itself but
everything around it: how the state is discretised, how ties between greedy actions are handled, and
how ε and α are annealed.

**Result: best average reward over 100 consecutive episodes of −140.52**, over 10,000 training
episodes. For reference, the environment is conventionally called "solved" at −110; the assignment
target was to stay above −150 repeatedly.

---

## What is implemented

**Expected SARSA update.** Instead of bootstrapping from a single sampled next action (SARSA) or
from the maximum (Q-learning), the target averages over all next actions weighted by their
probability under the current ε-greedy policy:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma\,\mathbb{E}_{\pi}\left[Q(s',\cdot)\right] - Q(s,a)\right]$$

This removes the variance contributed by sampling the next action, which shows up as visibly
smoother learning curves than SARSA at the same learning rate.

**State discretisation.** The continuous (position, velocity) pair is binned into a 2-D grid so a
tabular Q can be used at all. Bin count is the main lever on the bias–variance trade-off here: too
coarse and distinct states get merged, too fine and each cell is visited too rarely to learn.

**Correct tie-breaking.** Where several actions share the maximum Q-value, the greedy probability
mass (1 − ε) is split equally among all of them, in both the ε-greedy action selection and the
expectation inside the update. Getting this wrong silently biases the policy toward whichever action
`argmax` happens to return first — a common and hard-to-spot bug, since the agent still learns,
just worse.

**Annealing.** Both ε and α decay multiplicatively once per episode with a hard floor, so early
episodes explore and late episodes exploit while retaining a small permanent learning rate.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Runs on CPU; 10,000 episodes take a few minutes. Both the environment and ε-greedy selection are
stochastic, so the best-average-reward figure varies slightly between runs — a stable algorithm
should reproduce it within a small margin rather than exactly.

---

## Note

This originated as coursework for a taught postgraduate course and is published after assessment,
for reference and discussion. Please do not submit any part of it as your own work.
