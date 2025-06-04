from gymnasium.envs.registration import register


register(
    id="PassiveWalker-v0",
    entry_point="src.world.envs.PassiveWalkerGym:PassiveWalkerEnv",
    max_episode_steps=1000,
)


register(
    id="Ant_custom",
    entry_point="src.world.envs.AntCustomGym:AntCustomEnv",
    max_episode_steps=1000,
)


__version__ = "0.1"