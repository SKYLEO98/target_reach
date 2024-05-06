from gymnasium.envs.registration import register

register(
    id="target_reach-v0",
    entry_point="target_reach.envs:target_reach"
)
