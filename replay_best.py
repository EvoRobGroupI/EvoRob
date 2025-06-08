#!/usr/bin/env python3
"""
replay_best.py ──────────────────────────────────────────────────────────────
Locate the highest-fitness CMA-ES genotype and record a video of its rollout.
"""

import os
from pathlib import Path
import numpy as np
import imageio                         # pip install imageio
import mujoco                           # needed by TurtleGymEnv

# --- project-specific imports ----------------------------------------------
from src.utils.Filesys import get_project_root
from src.world.robot.controllers import MLP
from PPOvsEA_with_sense import TurtleGymEnv   # ← the env class you evolved in

# ---------------------------------------------------------------------------
# 1.  Find the best genotype under  results/TurtleWorld/CMAES/<gen>/
# ---------------------------------------------------------------------------

ROOT         = Path(get_project_root())
specific = ROOT / "results" / "TurtleWorld" / "test"
RESULTS_DIR  = specific / "CMAES"

print(RESULTS_DIR)
best_fitness = -np.inf
best_geno    = None
best_gen     = None

for gen_dir in sorted(RESULTS_DIR.iterdir()):
    f_file = gen_dir / "f_best.npy"
    x_file = gen_dir / "x_best.npy"
    if f_file.exists() and x_file.exists():
        f_val = float(np.load(f_file))
        if f_val > best_fitness:
            best_fitness = f_val
            best_geno    = np.load(x_file)
            best_gen     = gen_dir.name

if best_geno is None:
    raise RuntimeError(f"No `f_best.npy` / `x_best.npy` pair found under {RESULTS_DIR}")

print(f"✔ Best genotype found in generation {best_gen}  (fitness = {best_fitness:.2f})")

# ---------------------------------------------------------------------------
# 2.  Build controller → phenotype
# ---------------------------------------------------------------------------
# We need obs_dim and act_dim to build the same MLP as during evolution
_tmp_env      = TurtleGymEnv()                 # headless
obs_dim       = _tmp_env.observation_space.shape[0]
act_dim       = _tmp_env.action_space.shape[0]
controller    = MLP.NNController(obs_dim, act_dim)
controller.geno2pheno(best_geno)               # load weights

# ---------------------------------------------------------------------------
# 3.  Rollout & record
# ---------------------------------------------------------------------------
env          = TurtleGymEnv(render_mode="rgb_array", camera_name="topdown")
obs, _       = env.reset()
frames       = []

for _ in range(2_000):
    action         = np.clip(controller.get_action(obs), -1.0, 1.0)
    obs, _, done, _, _ = env.step(action)
    frames.append(env.render())                # 640×480 uint8 RGB
    if done:
        break

video_path = specific / "best_turtle.mp4"
imageio.mimsave(video_path, frames, fps=30)
print(f"🎞  saved video to {video_path.relative_to(ROOT)}")
