#!/usr/bin/env bash
# Sequential trainer. Running these concurrently made each ~4x slower through
# CPU contention (CIFAR hit 311s/epoch), so they run strictly one at a time.
set -u
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

echo "=== [1/4] Lunar Lander (Double DQN) ==="
( cd 06-lunar-lander-rl-agent && MAX_EPISODES=700 python lunar_lander_dqn.py > run_log.txt 2>&1 )
echo "lunar exit=$?"; grep -E "Best rolling|Mean return|landings|Solved \(" 06-lunar-lander-rl-agent/run_log.txt | head -5

echo "=== [2/4] Cart-Pole (DQN) ==="
( cd 05-cartpole-rl-agent && MAX_EPISODES=700 python cartpole_dqn.py > run_log.txt 2>&1 )
echo "cartpole exit=$?"; grep -E "Best rolling|Mean return|Perfect episodes|Solved \(" 05-cartpole-rl-agent/run_log.txt | head -5

echo "=== [3/4] Brain MRI CNN ==="
( cd 04-cancer-detection-mri && EPOCHS=25 python brain_tumor_mri.py > run_log.txt 2>&1 )
echo "mri exit=$?"; grep -E "Test accuracy|Sensitivity|Specificity|TUMOUR MISSED" 04-cancer-detection-mri/run_log.txt | head -6

echo "=== [4/4] CIFAR-10 CNN ==="
( cd 02-cifar10-image-classification-cnn && EPOCHS=15 BASELINE_EPOCHS=8 python cifar10_cnn.py > run_log.txt 2>&1 )
echo "cifar exit=$?"; grep -E "test accuracy|Dense baseline|CNN  " 02-cifar10-image-classification-cnn/run_log.txt | head -5

echo "=== ALL SEQUENTIAL RUNS FINISHED ==="
