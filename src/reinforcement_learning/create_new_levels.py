import os
import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

register(
    id="Sokoban-v2",
    entry_point="src.sokoban_env.sokoban_env:SokobanEnv",
    max_episode_steps=300
)

target_folder = "sokoban_levels"
env = gym.make('Sokoban-v2', render_mode="human")

game_no = 0
game_no_formatted = "{:05d}".format(game_no)

for _ in range(5000):
    while os.path.exists(f"{target_folder}/level_{game_no_formatted}.npy"):
        game_no += 1
        game_no_formatted = "{:05d}".format(game_no)

    env.reset()
    board, _, _, _, _ = env.info()
    np.save(f"{target_folder}/level_{game_no_formatted}.npy", board)
    print(f"File level_{game_no_formatted}.npy saved.")

print('done!')