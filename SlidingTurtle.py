# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv

from src.EA.CMAES import CMAES, CMAES_opts
from src.EA.NSGA import NSGAII, NSGA_opts
from src.utils.Filesys import get_project_root
from src.world.World import World  # base‑class wrapping Evo loops, RNG etc.
from src.world.robot.controllers import MLP

# -----------------------------------------------------------------------------
#  Constants & helper paths
# -----------------------------------------------------------------------------
ROOT_DIR = get_project_root()
ENV_NAME = "TurtleSlider"  # must be registered elsewhere in your code‑base

TURTLE_TEMPLATE_PATH = os.path.join(ROOT_DIR, "Turtle.xml")
SLIDER_TEMPLATE_PATH = os.path.join(ROOT_DIR, "SliderEnv.xml")

WORLD_OUT_PATH = os.path.join(ROOT_DIR, "TurtleSliderEnv.xml")
TURTLE_OUT_PATH = os.path.join(ROOT_DIR, "TurtleRobot.xml")

# -----------------------------------------------------------------------------
#  Turtle‑specific world implementation
# -----------------------------------------------------------------------------


def _scale_params(raw: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map *raw* genotype in **[−1, 1]** to a real value in **[lo, hi]**."""
    return lo + (raw + 1.0) * 0.5 * (hi - lo)


@dataclass
class MorphologySpec:
    """Container holding concrete (phenotypic) flipper segment lengths."""

    upper_L: float
    fore_L: float
    upper_R: float
    fore_R: float


class TurtleWorld(World):
    """Evolvable MuJoCo world that embeds a turtle inside the SliderEnv."""

    # ----------------------------- construction ----------------------------- #

    def __init__(self, n_repeats: int = 3, n_steps: int = 1_000):
        super().__init__()
        self.n_repeats = n_repeats
        self.n_steps = n_steps

        # Build *default* world once so we can frame the observation / action
        # dimensions and the MLP controller size.  (Lengths will be overwritten
        # for every individual later.)
        default_morph = MorphologySpec(0.15, 0.15, 0.15, 0.15)
        self._write_turtle_mjcf(default_morph)
        self._write_world_mjcf()

        with gym.make(ENV_NAME, robot_path=WORLD_OUT_PATH) as probe_env:
            self.state_space = probe_env.observation_space.shape[0]
            self.action_space = probe_env.action_space.shape[0]

        self.controller = MLP.NNController(self.state_space, self.action_space)
        self.n_weights = self.controller.n_params

        self.n_body_params = 4  # upper+fore L & R flippers
        self.n_params = self.n_weights + self.n_body_params

    # --------------------------- geno → pheno ------------------------------ #

    def _map_genotype(self, genotype: np.ndarray) -> tuple[MorphologySpec, np.ndarray]:
        """Split genotype into morphology (4) and controller weights."""
        assert len(genotype) == self.n_params, "Genotype length mismatch"
        body_raw, ctrl_raw = genotype[:self.n_body_params], genotype[self.n_body_params:]
        #  Scale flipper lengths roughly from 5 cm → 25 cm.
        lengths = _scale_params(body_raw, lo=0.05, hi=0.25)
        morph = MorphologySpec(*lengths)
        return morph, ctrl_raw

    # --------------------------- MJCF generation --------------------------- #

    def _write_turtle_mjcf(self, morph: MorphologySpec) -> None:
        """Clone *Turtle copy.xml* and overwrite flipper lengths/positions."""
        tree = ET.parse(TURTLE_TEMPLATE_PATH)
        root = tree.getroot()

        # ------------------------------------------------------------------------------------------------------------------
        # Helper to overwrite a limb pair (upper + fore geom + fore body‑pos).
        # ------------------------------------------------------------------------------------------------------------------
        def _overwrite_side(side: str, upper_len: float, fore_len: float) -> None:
            # Upper flipper geom …
            geom_upper = root.find(f".//geom[@name='flipper_{side}_upper_geom']")
            geom_upper.set("fromto", f"0 0 0  {upper_len:.3f} 0 0")
            # Forearm body anchor …
            body_fore = root.find(f".//body[@name='flipper_{side}_fore']")
            body_fore.set("pos", f"{upper_len:.3f} 0 0")
            # Forearm geom …
            geom_fore = root.find(f".//geom[@name='flipper_{side}_fore_geom']")
            geom_fore.set("fromto", f"0 0 0  {fore_len:.3f} 0 0")

        _overwrite_side("L", morph.upper_L, morph.fore_L)
        _overwrite_side("R", morph.upper_R, morph.fore_R)

        tree.write(TURTLE_OUT_PATH)

    def _write_world_mjcf(self) -> None:
        """Compose SliderEnv + (include) Turtle."""
        world_tree = ET.parse(SLIDER_TEMPLATE_PATH)
        world_root = world_tree.getroot()

        # Append a top‑level <include …> – this is the same trick used in the
        # Ant example.
        world_root.append(ET.Element("include", attrib={"file": os.path.basename(TURTLE_OUT_PATH)}))
        world_tree.write(WORLD_OUT_PATH)

    # ------------------------------ evaluation ----------------------------- #

    def evaluate_individual(self, genotype: np.ndarray) -> tuple[float, np.ndarray]:
        """Run ⟨n_repeats⟩ rollouts and return mean reward(s)."""
        morph, ctrl_weights = self._map_genotype(genotype)
        self.controller.geno2pheno(ctrl_weights)

        # Regenerate MJCFs for this morphology.
        self._write_turtle_mjcf(morph)
        self._write_world_mjcf()

        envs = AsyncVectorEnv([
            lambda _=i: gym.make(
                ENV_NAME,
                robot_path=WORLD_OUT_PATH,
                reset_noise_scale=0.1,
                max_episode_steps=self.n_steps,
            )
            for i in range(self.n_repeats)
        ])

        rewards_buf = np.zeros((self.n_steps, self.n_repeats))
        multi_rewards_buf = np.zeros((self.n_steps, self.n_repeats, 2))  #   ↳ (forward, −ctrl‑cost)

        observations, _ = envs.reset()
        done_mask = np.zeros(self.n_repeats, dtype=bool)

        for step in range(self.n_steps):
            actions = np.where(done_mask[:, None], 0.0, self.controller.get_action(observations.T).T)
            observations, rewards, dones, truncated, infos = envs.step(actions)

            active = ~done_mask
            rewards_buf[step, active] = rewards[active]

            # Expect the custom env to populate info dict like the Ant one.
            forward = np.array(infos["reward_forward"])
            ctrl = np.array(infos["ctrl_cost"])
            multi_rewards_buf[step, active] = np.vstack([forward, -ctrl]).T[active]

            done_mask |= dones | truncated
            if np.all(done_mask):
                break

        envs.close()
        mono_score = rewards_buf.sum(axis=0).mean()
        multi_score = multi_rewards_buf.sum(axis=0).mean(axis=0)
        return mono_score, multi_score

    # --------------------------------------------------------------------- #


# -----------------------------------------------------------------------------
#  Evolutionary helpers (identical to the Ant version, just renamed world)
# -----------------------------------------------------------------------------

def _run_EA_single(ea: CMAES, world: TurtleWorld) -> None:
    for _ in range(ea.n_gen):
        pop = ea.ask()
        fits = np.fromiter((world.evaluate_individual(g)[0] for g in pop), dtype=float)
        ea.tell(pop, fits)


def _run_EA_multi(ea: NSGAII, world: TurtleWorld) -> None:
    for _ in range(ea.n_gen):
        pop = ea.ask()
        fits = np.vstack([world.evaluate_individual(g)[1] for g in pop])
        ea.tell(pop, fits)


# -----------------------------------------------------------------------------
#  Convenience utilities (visualisation, video …) – unchanged
# -----------------------------------------------------------------------------

def _render_best(world: TurtleWorld, video_fname: str = "EvoTurtle_best.mp4", steps: int = 1_000):
    env = gym.make(ENV_NAME, robot_path=WORLD_OUT_PATH, render_mode="rgb_array")
    obs, _ = env.reset()
    frames, rewards = [], []

    for _ in range(steps):
        frames.append(env.render())
        obs, r, term, trunc, _ = env.step(world.controller.get_action(obs))
        rewards.append(r)
        if term or trunc:
            break

    env.close()
    print("Episode return:", np.sum(rewards))

    import imageio

    imageio.mimsave(video_fname, frames, fps=30)


# -----------------------------------------------------------------------------
#  Main entry point
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: C901  (complexity fine for script)
    # 1⃣  Smoke‑test a random genotype ───────────────────────────────────────
    world = TurtleWorld()
    dummy = np.random.uniform(-1, 1, world.n_params)
    world.evaluate_individual(dummy)  # does not raise ⇒ pipeline wired up

    pop_size = 250

    # 3⃣  Multi‑objective optimisation (NSGA‑II) ─────────────────────────────
    NSGA_opts.update({
        "min": -1.0,
        "max": 1.0,
        "num_parents": pop_size,
        "num_generations": 100,
        "mutation_prob": 0.3,
        "crossover_prob": 0.5,
    })

    results_dir = os.path.join(ROOT_DIR, "results", ENV_NAME, "multi")
    os.makedirs(results_dir, exist_ok=True)

    ea_multi = NSGAII(pop_size, world.n_params, NSGA_opts, results_dir)
    _run_EA_multi(ea_multi, world)

    # 4⃣  Visualise best individual ─────────────────────────────────────────
    best = np.load(os.path.join(results_dir, "99", "x_best.npy"))
    world.evaluate_individual(best)  # to set controller weights + morphology
    _render_best(world)


if __name__ == "__main__":
    main()
