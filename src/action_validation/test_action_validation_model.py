# This files tests the performance of the action_validation_model
import glob
import random

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import register

from src.action_validation.action_validatoin_model import ActionValidationModel
from src.common.model_actions_to_env_actions import MODEL_ACTIONS_TO_ENV_ACTIONS
from src.common.preprocess_image import preprocess
from src.frame_prediction.frame_prediction_model import FramePredictionModel

num_actions = 5
env_model = FramePredictionModel(3, num_actions)
env_model.load_state_dict(torch.load('res/frame_prediction/frame_prediction_model.pth'))
env_model.cuda()


model = ActionValidationModel()
model.cuda()

######################## validation #################################3

register(
    id="Sokoban-v2",
    entry_point="src.sokoban_env.sokoban_env:SokobanEnv",
    max_episode_steps=300
)


def calculate_accuracy(pred, actual, cut_off = 0.5):
    predicted = (torch.nn.Sigmoid()(pred) > cut_off).float()
    return (predicted == actual).float().mean()


class SokobanEnvWrapper:
    def __init__(self):
        self.env = gym.make('Sokoban-v2', render_mode="rgb_array", reset=False)
        self.row_count = 160
        self.column_count = 160
        self.action_size = 5

    def start_random_level_validate(self):
        level_no = random.randint(4800, 5061)
        level_counter_formatted = "{:05d}".format(level_no)

        moves_in_level = sorted(glob.glob(f"sokoban_levels_validate/level_{level_counter_formatted}*.npy"))
        start = moves_in_level[0]
        state = np.load(start)
        start_state = np.load(moves_in_level[0])

        start_state, state = self.random_symmetry(start_state, state)
        self.env.reset(options={'state': state, 'start_state': start_state })
        return [start_state, state]


    def random_symmetry(self, state1, state2):
        rotation = random.randint(0, 3)
        for _ in range(rotation):
            state1 = np.rot90(state1)
            state2 = np.rot90(state2)
        if random.randint(0, 1) == 1:
            state1 = np.fliplr(state1)
            state2 = np.fliplr(state2)
        return state1, state2

    def get_next_state(self, state, action):
        self.env.reset(options={ 'state': state[1], 'start_state': state[0] } )
        observation, reward, terminated, truncated, info = self.env.step(MODEL_ACTIONS_TO_ENV_ACTIONS[action])
        return [state[0], observation], info["action.moved_player"] or info["action.moved_box"]

    def get_encoded_state(self, state):
        self.env.reset(options={'state': state[0]})
        initial_state_render = self.env.render()
        self.env.reset(options={ 'state': state[1], 'start_state': state[0] })
        current_render = self.env.render()
        return initial_state_render, current_render

    def render_state(self, state):
        self.env.reset(options={ 'state': state[1], 'start_state': state[0] })
        return self.env.render()


env = SokobanEnvWrapper()
state = None
cutoff = 0.5
total_count = 10000

model.load_state_dict(torch.load(f'res/action_validator/action_validation_model.pth'))
model.eval()
argmax = -1
max_value = 0
correct = 0
false_positive = 0
false_negative = 0
total_positive = 0
for idx in range(total_count):
    state = env.start_random_level_validate()
    action = random.randint(0, 3)
    one_hot_action = torch.nn.functional.one_hot(torch.tensor(action), num_classes=num_actions).cuda()
    start_image_np, current_image_np = env.get_encoded_state(state)
    start_image = preprocess(start_image_np).unsqueeze(0)
    current_image = preprocess(current_image_np).unsqueeze(0)
    predicted_frame_change = env_model(start_image, current_image, one_hot_action.unsqueeze(0))
    follow_up_state, is_valid = env.get_next_state(state, action)
    plausibility_prediction = model(current_image, predicted_frame_change + current_image)
    acc = calculate_accuracy(plausibility_prediction, is_valid, cutoff).cpu().detach().item()

    if is_valid:
        total_positive += 1
    if acc < 1 and not is_valid:
        false_positive += 1
    if acc < 1 and is_valid:
        false_negative += 1
    if acc >= 1:
        correct += 1

print(
    f"correct: {correct}, false positive: {false_positive}/{total_count - total_positive}. false negative: {false_negative}/{total_positive}")
