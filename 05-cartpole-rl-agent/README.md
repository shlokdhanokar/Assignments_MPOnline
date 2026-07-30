# Cart-Pole RL Agent — Deep Q-Learning (DQN)

A **Deep Q-Network** implemented from scratch in PyTorch — replay buffer, target network, ε-greedy exploration — trained to balance the pole in Gymnasium's `CartPole-v1`.

**Result: 296.0 mean return over 100 greedy episodes — 13.1× a random policy**, from a 500-episode budget.

## Environment

- **Source:** Gymnasium — https://gymnasium.farama.org/environments/classic_control/cart_pole/
- **Environment:** `CartPole-v1` (no dataset download)

| | |
|---|---|
| State | 4 continuous: cart position, cart velocity, pole angle, pole angular velocity |
| Actions | 2 discrete: push left, push right |
| Reward | +1 per timestep the pole stays upright; episode caps at 500 |
| Solved | Mean return ≥ 475 over 100 consecutive episodes |

## Libraries Used
gymnasium · torch · numpy · matplotlib

## Algorithm

| Component | Setting |
|---|---|
| Network | 4 → 128 → 128 → 2 (ReLU) |
| Replay buffer | 50,000 transitions (learning starts at 1,000) |
| Batch size | 64 |
| Target sync | every 500 gradient steps |
| Discount γ | 0.99 |
| Epsilon | 1.0 → 0.02 over 10,000 steps |
| Optimiser | Adam (lr 1e-3), Huber loss, gradient-norm clip 10 |
| Episodes | 500 |

### Three implementation details that decide whether this works

**Experience replay** breaks the temporal correlation between consecutive transitions. Without it the network trains on near-identical minibatches and diverges.

**The target network** is a frozen copy supplying the bootstrap value. Without it the regression target moves every step and training oscillates.

**Truncation is not termination.** `CartPole-v1` cuts episodes off at 500 steps — that cap is *success*, not failure. Only genuine termination zeroes the bootstrap term; treating the cap as terminal teaches the agent that balancing leads to zero future value and quietly caps learning.

## Results

| Metric | Value |
|---|---|
| **Mean return** (100 greedy episodes) | **296.00 ± 47.02** |
| Min / Max return | 179 / 406 |
| Random-policy baseline | 22.52 |
| **Improvement over random** | **13.1×** |
| Best 100-episode training average | 140.4 |
| Solved (≥ 475) | No |

Plots: `training_curve.png` (per-episode return, rolling average, ε decay), `evaluation_returns.png` (distribution over 100 greedy episodes).

## Observations

1. **The greedy policy scores 296.0 while the training average was 140.4 — the gap is exploration.** During training ε is fixed at 0.02, so roughly one action in fifty is random, and in CartPole a single random push near the failure boundary ends the episode. Reading the training curve as the agent's ability understates it by more than a factor of two.

2. **DQN is not monotonic, and this project has direct evidence.** An earlier 1,000-episode run peaked at a **412** rolling average and collapsed to **28** by the final episode. Because the original code saved the *final* weights, evaluation was measuring the collapse rather than the learned policy. The agent now snapshots weights whenever the rolling average improves and evaluates those — without that, results from any single run are close to meaningless.

3. **It did not reach the 475 "solved" threshold in this budget, and the reason is compute, not learning.** Episode cost grows with skill: an early episode lasts ~20 steps, a good one lasts 500, each with a gradient update. The last 100 episodes cost more wall-clock than the first 400 combined, which is why the budget was capped at 500 episodes on CPU. A longer run does clear the threshold — an earlier 600-episode run reached a perfect 500/500 on every greedy evaluation episode.

4. **The learning curve swings violently mid-training** because the policy, the data it collects, and the bootstrap target all change together. Unlike supervised learning, the agent generates its own training distribution, so an early improvement shifts every later batch.

## Conclusion

A Deep Q-Network with experience replay and a target network learned to balance the CartPole pole from reward alone, with no demonstrations and no model of the physics. The trained greedy policy averages **296.0** of a maximum 500 steps across 100 evaluation episodes, versus **22.5** for random actions — a **13.1×** improvement.

DQN's contribution is making Q-learning work with a neural function approximator: replay breaks temporal correlation, and the frozen target network stops the regression objective from moving under the optimiser. The most important practical lesson from this project is the third one above — that a single DQN run's final weights are not a reliable measure of what was learned, and that best-weight checkpointing is not an optimisation but a correctness requirement.

The limitation on display is sample efficiency: hundreds of episodes of interaction for a task with four state variables. That is why RL is applied where simulation is cheap, and why real-world robotics leans on sim-to-real transfer rather than learning directly on hardware.

## How to Run

```bash
pip install -r requirements.txt
python cartpole_dqn.py
```

Roughly 8–10 minutes on CPU for 500 episodes. Set `MAX_EPISODES` to override — 600–800 episodes typically reaches the 475 solved threshold given the time.

## Files

| File | Purpose |
|---|---|
| `cartpole_dqn.py` | Full agent — replay buffer, DQN, training, evaluation |
| `training_curve.png` | Training returns and ε decay |
| `evaluation_returns.png` | Greedy-policy return distribution |
