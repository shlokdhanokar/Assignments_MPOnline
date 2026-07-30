# Lunar Lander RL Agent — Double DQN

A **Double Deep Q-Network** implemented from scratch in PyTorch that learns to land a spacecraft in Gymnasium's `LunarLander-v3` from reward alone — no demonstrations, no model of the physics.

**Result: solved.** 221.4 mean return over 100 greedy evaluation episodes (threshold is 200), with **81/100 successful landings and zero crashes**.

## Environment

- **Source:** Gymnasium — https://gymnasium.farama.org/environments/box2d/lunar_lander/
- **Environment:** `LunarLander-v3` (Box2D physics simulation — no dataset download)

| | |
|---|---|
| State | 8 continuous variables: x, y, vx, vy, angle, angular velocity, left-leg contact, right-leg contact |
| Actions | 4 discrete: do nothing, fire left engine, fire main engine, fire right engine |
| Reward | Shaped by distance/speed/tilt · −0.3 per main-engine frame · +10 per leg contact · **+100 landing** · **−100 crash** |
| Solved | Mean return ≥ 200 over 100 consecutive episodes |

## Libraries Used
gymnasium[box2d] · torch · numpy · matplotlib

## Algorithm

Double DQN, chosen deliberately over vanilla DQN. Plain DQN takes a `max` over the target network's own estimates, which systematically **over-estimates** action values. In this environment that bias inflates the value of firing engines and produces agents that hover indefinitely instead of committing to a descent. Double DQN separates the two roles — the **online** network chooses the next action, the **target** network scores it.

| Component | Setting |
|---|---|
| Network | 8 → 128 → 128 → 4 (ReLU), 18,180 parameters |
| Replay buffer | 100,000 transitions (learning starts at 5,000) |
| Batch / update | 64, one gradient step per 4 environment steps |
| Target sync | every 1,000 gradient steps |
| Discount γ | 0.99 |
| Epsilon | 1.0 → 0.02 over 60,000 steps |
| Optimiser | Adam (lr 5e-4), Huber loss, gradient-norm clip 10 |
| Episodes | 700 |

### Three implementation details that decide whether this works

**Truncation is not termination.** Episodes are cut off at 1,000 steps. That cap is not failure, so only genuine termination (landed or crashed) zeroes the bootstrap term. Treating the cap as terminal teaches the agent that surviving leads to zero future value.

**Evaluate the best weights, not the final ones.** DQN is not monotonic — a run can reach a good policy and then collapse. In an earlier CartPole run in this repository the rolling average peaked at 412 and fell to 28 by the final episode. This agent snapshots weights whenever the 100-episode rolling average improves and evaluates *those*.

**Exploration is disabled at evaluation.** Training-time returns are depressed by the 2% random-action rate; the greedy policy is what the agent has actually learned.

## Results

### Evaluation — 100 episodes, greedy policy

| Metric | Value |
|---|---|
| **Mean return** | **221.43 ± 65.01** |
| Min / Max return | −26.9 / 320.7 |
| **Successful landings (≥ 200)** | **81 / 100** |
| **Crashes (≤ −100)** | **0 / 100** |
| Random-policy baseline | −202.3 |
| **Solved (≥ 200)** | **YES** |

### Training trajectory

| Episode | 100-episode average | ε |
|---|---|---|
| 100 | −176.2 | 0.848 |
| 300 | −75.7 | 0.020 |
| 500 | 6.3 | 0.020 |
| 700 | 147.9 | 0.020 |
| Best | **154.9** | |

Plot: `training_curve.png` (per-episode return with rolling average, and the evaluation distribution).

## Observations

1. **The agent went from −202 (random) to +221, and lands 81% of the time without a single crash in 100 episodes.** Zero crashes is the more telling number than the mean: the policy learned to be safe before it learned to be precise. The 19 non-landings are slow or off-pad touchdowns, not failures.

2. **Training average (154.9) understates the policy (221.4) by ~67 points.** That entire gap is exploration. With ε fixed at 0.02, roughly one action in fifty is random, and in a task where a single mistimed burn near the ground ruins a descent, that is enough to drag the training curve well below what the greedy policy achieves. Reading the training curve as the agent's ability would have declared this run unsolved.

3. **Learning is distinctly two-staged, visible in the trajectory above.** Returns sit near −176 while the agent learns to stop crashing, plateau near zero around episode 500 as it discovers hovering avoids the −100 penalty, and only then climb toward +200 as it works out that hovering forfeits the +100 landing bonus. The plateau at zero is the agent solving the wrong problem correctly.

4. **This is much harder than CartPole despite both being "small" control tasks.** CartPole has 4 state variables, 2 actions, and reward that rises monotonically with survival. Lunar Lander has 8 variables, 4 actions, and a reward function containing a genuine conflict — every main-engine frame costs fuel now against a landing bonus later. The agent must learn to spend reward to earn more of it.

## Conclusion

A Double DQN agent learned to land the Lunar Lander from reward alone, reaching **221.4 mean return** over 100 greedy evaluation episodes against **−202.3** for random actions, with **81/100 successful landings and no crashes** — clearing the 200-point solved threshold.

Three ingredients carry the training: experience replay to decorrelate consecutive transitions, a frozen target network to stop the regression objective moving under the optimiser, and decayed ε-greedy exploration. Double Q-learning is the addition that matters at this difficulty, removing the over-estimation bias that otherwise teaches the agent to hover rather than land.

The cost on display is sample efficiency: roughly 700 episodes and hundreds of thousands of simulated frames for a task a human learns in minutes. That is why reinforcement learning is applied where simulation is cheap, and why real-world robotics leans on sim-to-real transfer rather than learning on hardware.

## How to Run

```bash
pip install -r requirements.txt
python lunar_lander_dqn.py
```

Roughly 25–35 minutes on CPU for 700 episodes. Set `MAX_EPISODES` / `EVAL_EPISODES` to override. Saves `lunar_lander_dqn.pt` and `training_curve.png`.

> `gymnasium[box2d]` needs a C++ toolchain on some systems. If the Box2D wheel fails to build, install `swig` first.

## Files

| File | Purpose |
|---|---|
| `lunar_lander_dqn.py` | Full agent — replay buffer, Double DQN, training, evaluation |
| `training_curve.png` | Training returns and evaluation distribution |
