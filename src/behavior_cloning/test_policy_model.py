import os
import random
import gymnasium as gym
import torch
from PIL import Image
from gymnasium.envs.registration import register
import numpy as np

from src.common.encoder_resnet_model import EncoderResnetModel
from src.common.image_folders import synthetic_validation_folders
from src.common.model_actions_to_env_actions import MODEL_ACTIONS_TO_ENV_ACTIONS
from src.common.preprocess_image import preprocess

register(
    id="Sokoban-v2",
    entry_point="src.sokoban_env.sokoban_env:SokobanEnv",
    max_episode_steps=300
)

target_folder = "sokoban_solutions"
env = gym.make('Sokoban-v2', render_mode="rgb_array")

observation, _ = env.reset()

ACTION_LOOKUP = {
    1: 'push up',
    2: 'push down',
    3: 'push left',
    4: 'push right'
}

action_letter = {
    0: 'l',
    1: 'u',
    2: 'r',
    3: 'd'
}

model = EncoderResnetModel(5).cuda()
model.load_state_dict(torch.load(f'res/behavior_cloning/policy_model_synthetic.pth'))
model.eval()


def times_visited(obs, observations):
    counter = -1
    for observation in observations:
        if np.min(observation == obs):
            counter += 1
    return counter


level_counter = 0
level_counter_formatted = "{:05d}".format(level_counter)
while os.path.exists(f"sokoban_levels_validate/level_{level_counter_formatted}.npy"):
    level_counter += 1
    level_counter_formatted = "{:05d}".format(level_counter)
level_counter -= 1


def start_random_level(env, game_no=None):
    level_no = game_no if game_no is not None else random.randint(4800, 5061)
    level_counter_formatted = "{:05d}".format(level_no)
    start_state = np.load(f"sokoban_levels_validate/level_{level_counter_formatted}.npy")
    env.reset(options={'state': start_state})
    return start_state, np.copy(start_state)


solved = 0
num_puzzles = 5061 - 4800
mean_loss = 0
for idx in range(4800, 5061):
    print(f"{idx}")
    observations = []
    observation, _ = start_random_level(env, idx)
    observations.append(np.copy(observation))
    start_img = env.render()
    for step in range(100):
        img = env.render()

        input_tensor1 = preprocess(start_img)
        input_tensor2 = preprocess(img)

        action = model(input_tensor1.unsqueeze(0), input_tensor2.unsqueeze(0))

        visited_count = times_visited(observations[-1], observations)
        if visited_count == 0:
            env_action = MODEL_ACTIONS_TO_ENV_ACTIONS[torch.max(action, dim=0).indices.item()]
        else:
            if visited_count >= 4:
                break
            next_action = torch.topk(action, visited_count + 1, dim=0).indices[-1].item()
            env_action = MODEL_ACTIONS_TO_ENV_ACTIONS[next_action]
        observation, reward, terminated, truncated, info = env.step(env_action)
        observations.append(np.copy(observation))

        if terminated > 0:
            if reward > 10: # game won
                solved += 1
                print(f"solved: {solved}")
            break

print(f"done. solved {solved} / {num_puzzles} puzzles")


#####################################
# shows the results for a single level
####################################
import glob

action_letter_map = {
    0: 'l',
    1: 'u',
    2: 'r',
    3: 'd'
}

prefix = f"{synthetic_validation_folders[0]}/game_00000"
files = sorted(glob.glob(f"{prefix}*.png"))

f = open(f"{prefix}.txt", "r")
solution = f.readline().split()
f.close()

start_frame = Image.open(files[0])
correct = 0
for idx, file in enumerate(files[:-1]):
    current_frame = Image.open(file)

    input_tensor1 = preprocess(start_frame)
    input_tensor2 = preprocess(current_frame)

    action = model(input_tensor1.unsqueeze(0), input_tensor2.unsqueeze(0))
    action_letter = action_letter_map[torch.max(action, dim=0).indices.item()]
    if action_letter == solution[idx]:
        correct += 1
    print(f"True action: {solution[idx]}. Policy action: {action_letter}")
print(f"Correct {correct}. Total {len(files)}")