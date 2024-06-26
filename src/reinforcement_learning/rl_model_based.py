'''
This code is adapted from
https://github.com/foersterrobert/AlphaZeroFromScratch
Credit goes to the original author
'''


import gc
import os
import time

import cv2
import numpy as np
from torch import nn
from torchvision import transforms

from src.common.encoder_resnet_model import EncoderResnetModel
from src.sokoban_simulator.simulator import Simulator

print(np.__version__)

import torch

print(torch.__version__)

torch.manual_seed(0)

import random
import math
import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id="Sokoban-v2",
    entry_point="src.sokoban_env.sokoban_env:SokobanEnv",
    max_episode_steps=300
)

some_test_level =  np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 2, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 4, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 5, 1, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

def print_gpu_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                print(type(obj), obj.size())
        except Exception as e:
            pass
    print(len(gc.get_objects()))
    print("_______________________________")


class SokobanEnvWrapper:
    def __init__(self):
        self.sim = Simulator()
        self.env = gym.make('Sokoban-v2', render_mode="rgb_array", reset=False)
        self.start_random_level(0)
        self.action_size = 5
        self.level_counter = 0
        level_counter_formatted = "{:05d}".format(self.level_counter)
        while os.path.exists(f"sokoban_levels/level_{level_counter_formatted}.npy"):
            self.level_counter += 1
            level_counter_formatted = "{:05d}".format(self.level_counter)
        self.level_counter -= 1
        print(f"levels found: {self.level_counter}")

    def start_random_level(self, game_no=None):
        level_no = game_no if game_no is not None else random.randint(0, self.level_counter)
        level_counter_formatted = "{:05d}".format(level_no)
        start_state = np.load(f"sokoban_levels/level_{level_counter_formatted}.npy")
        start_state, _ = self.random_symmetry(start_state, start_state)
        self.env.reset(options={'state': start_state})
        start = self.env.render()
        preprocess = transforms.Compose([
            transforms.ToTensor()
        ])
        input_tensor1 = preprocess(start)
        input_tensor1 -= 0.5
        start_frame = input_tensor1
        current_frame = input_tensor1
        return torch.concatenate([start_frame, current_frame], dim=0).detach().cpu().numpy()

    def random_symmetry(self, state1, state2):
        rotation = random.randint(0, 3)
        for _ in range(rotation):
            state1 = np.rot90(state1)
            state2 = np.rot90(state2)
        if random.randint(0, 1) == 1:
            state1 = np.fliplr(state1)
            state2 = np.fliplr(state2)
        return state1, state2

    def get_next_states(self, states, actions):
        state_tensor = torch.tensor(states, device='cuda')
        if len(state_tensor.shape) == 3:
            state_tensor = state_tensor.unsqueeze(0)
        current_states, next_states = state_tensor[:, :3, :, :], state_tensor[:, 3:, ::]
        one_hot_tensors = torch.zeros((len(actions), 5), device="cuda")
        for i, num in enumerate(actions):
            one_hot_tensors[i, num] = 1
        next_states, is_valid, terminal = self.sim.step_batch(current_states, next_states, one_hot_tensors)
        terminal = torch.nn.Sigmoid()(terminal) > 0.5
        is_valid = torch.nn.Sigmoid()(is_valid) > 0.5
        return np.concatenate([states[:, :3, :, :], next_states.detach().cpu().numpy()], axis=1), is_valid.detach().cpu().numpy(), terminal.detach().cpu().numpy()

    def apply_actions_to_state(self, state, actions):
        batch = np.stack([state] * len(actions), axis=0)
        return self.get_next_states(batch, actions)

    def get_values_and_terminated(self, states):
        state_tensor = torch.tensor(states, device='cuda')
        terminal = (torch.nn.Sigmoid()(self.sim.is_terminated(state_tensor[:, :3, :, :].cuda(), state_tensor[:, 3:, :, :].cuda())) > 0.5).detach().cpu().numpy()
        return 1 * terminal, terminal

    def render_to_file(self, state, filename):
        start_frame = (np.moveaxis(state[:3, :, :], 0, -1) + 0.5) * 255
        next_frame = (np.moveaxis(state[3:, :, :], 0, -1) + 0.5) * 255
        cv2.imwrite(f"tmp/{filename}-next.png", cv2.cvtColor(next_frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"tmp/{filename}-start.png", cv2.cvtColor(start_frame, cv2.COLOR_RGB2BGR))


gamma = 0.95
class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []

        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def remove_from_parent(self):
        parent = self.parent
        if parent is not None:
            parent.children.remove(self)
            if len(parent.children) == 0:
                parent.remove_from_parent()

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = (self.value_sum / self.visit_count)
        else:
            q_value = (child.value_sum / child.visit_count)
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self, policy):
        child_states, move_valids, terminals = self.game.apply_actions_to_state(self.state, [0, 1, 2, 3])
        for action, prob in enumerate(policy):
            if action == 4:
                continue
            if prob > 0:
                child_state, move_valid, terminal = child_states[action], move_valids[action].item(), terminals[action].item()
                if move_valid:
                    child = Node(self.game, self.args, child_state, self, action, prob)
                    self.children.append(child)
        return []

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(value * gamma)


class MCTSParallel:
    def __init__(self, game, args, policy_model, value_model):
        self.game = game
        self.args = args
        self.policy_model = policy_model
        self.value_model = value_model

    @torch.no_grad()
    def search(self, states, spGames):
        if spGames[0].root is None:
            policy = self.policy_model(torch.tensor(states, device='cuda'))
            if len(policy.shape) == 1:
                policy = policy.unsqueeze(0)
            policy = torch.softmax(policy, axis=1).cpu().numpy()
            policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
                     * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])

            for i, spg in enumerate(spGames):
                spg_policy = policy[i]
                spg_policy[4] = 0.0  # no lazy action!
                spg_policy /= np.sum(spg_policy)

                spg.root = Node(self.game, self.args, states[i], visit_count=1)

        for search in range(self.args['num_searches']):
            nodes = []
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()
                nodes.append(node)

            states = np.stack([node.state for node in nodes])
            values, is_terminals = self.game.get_values_and_terminated(states)

            for idx, spg in enumerate(spGames):
                value, is_terminal = values[idx].item(), is_terminals[idx].item()

                if is_terminal:
                    nodes[idx].backpropagate(value * 10) # make sure to use path that terminates

                else:
                    spg.node = nodes[idx]

            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if
                                  spGames[mappingIdx].node is not None]

            if len(expandable_spGames) > 0:
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])

                policy = self.policy_model(torch.tensor(states, dtype=torch.float32, device="cuda"))
                value = self.value_model(torch.tensor(states, dtype=torch.float32, device="cuda"))
                if len(policy.shape) == 1:
                    policy = policy.unsqueeze(0)
                policy = policy * torch.tensor([1, 1, 1, 1, 0], device="cuda")
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
                     * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])
                value = value.cpu().numpy()

            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]
                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)
                node.backpropagate(spg_value)


class AlphaZeroParallel:
    def __init__(self, policy_model, policy_optimizer, value_model, value_optimizer, game, args):
        self.policy_model = policy_model
        self.policy_optimizer = policy_optimizer
        self.value_model = value_model
        self.value_optimizer = value_optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, policy_model, value_model)


    def selfPlay(self, eval_only=False):
        return_memory = []
        spGames = [SPG(self.game, validation_set = eval_only) for spg in range(self.args['num_parallel_games'])]
        steps = 0
        max_steps = self.args['max_steps_per_level']
        won_games = 0

        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])

            self.mcts.search(states, spGames)

            actions = []
            for i in range(len(spGames)):
                spg = spGames[i]

                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                valid_moves = np.array([1.0, 1.0, 1.0, 1.0, 0.0]) # exclude lazy action
                action_probs = action_probs * valid_moves
                if np.sum(action_probs) > 0:
                    action_probs /= np.sum(action_probs)
                else:
                    action_probs = valid_moves / np.sum(valid_moves)

                action = np.random.choice(self.game.action_size, p=action_probs)
                one_hot_vector = np.zeros(self.game.action_size, dtype=int)
                one_hot_vector[action] = 1

                spg.memory.append((spg.root.state, one_hot_vector))
                actions.append(action)

            next_states, is_valids, is_terminals = self.game.get_next_states(states, actions)

            spGames_to_delete = []
            for i in range(len(spGames)):
                spg = spGames[i]
                if is_valids[i].item():
                    spg.state, is_terminal = next_states[i], is_terminals[i].item()

                    value = 1.0 if is_terminal else 0.0

                    if is_terminal:
                        self.game.render_to_file(spg.state, "won-" + str(i))

                    root_update_success = spg.update_root(actions[i])
                    if is_terminal or steps > max_steps or not root_update_success:
                        if is_terminal and value > 0:
                            won_games += 1
                        for idx, (hist_neutral_state, hist_action_probs) in enumerate(spg.memory):
                            gamma = 0.95
                            return_memory.append((
                                hist_neutral_state,
                                hist_action_probs,
                                value * gamma ** (steps - idx)
                            ))
                        spGames_to_delete.append(i)
                else:
                    print("invalid! "  + str(steps) + " " + str(i))
                    self.game.render_to_file(next_states[i], "invalid_" + str(i) + " " + str(steps))
                    for idx, (hist_neutral_state, hist_action_probs) in enumerate(spg.memory):
                        return_memory.append((
                            hist_neutral_state,
                            hist_action_probs,
                            0
                        ))
                    spGames_to_delete.append(i)

            for i in spGames_to_delete[::-1]:
                del spGames[i]

            steps += 1
            gc.collect()
            torch.cuda.empty_cache()

        return return_memory, won_games

    def train(self, memory):
        random.shuffle(memory)
        mean_policy_loss = 0
        mean_value_loss = 0
        counter = 0
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:batchIdx + self.args['batch_size']]
            state, selected_actions, value_targets = zip(*sample)
            selected_actions, state, value_targets = np.array(selected_actions), np.array(state), np.array(
                value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device="cuda")
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device="cuda")

            out_policy = self.policy_model(state)
            if len(out_policy.shape) == 1:
                out_policy = out_policy.unsqueeze(0)

            mask = value_targets.squeeze() > 0
            policy_targets = torch.tensor(selected_actions, dtype=torch.float32, device="cuda")[mask]
            out_policy = out_policy[mask]

            if policy_targets.shape[0] > 0:
                policy_loss = torch.nn.CrossEntropyLoss()(out_policy, policy_targets)
                mean_policy_loss += policy_loss.item()
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

            out_value = self.value_model(state)
            if len(out_value.shape) == 1:
                out_value = out_value.unsqueeze(0)
            value_loss = torch.nn.MSELoss()(out_value, value_targets)
            mean_value_loss += value_loss.item()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            counter += 1

        print(f"policy loss {mean_policy_loss / counter}, value loss {mean_value_loss / counter}")


    def learn(self):
        won_games_total = 0
        games_played_total = 0
        win_rates = []
        best = 0
        last_wind_rates = []
        for iteration in range(0, self.args['num_iterations']):
            memory = []
            won_games = 0
            self.policy_model.eval()
            self.value_model.eval()
            start_time = time.time()
            with torch.no_grad():
                memory_new, won_matches = self.selfPlay()
            memory += memory_new
            won_games += won_matches
            print(f"game solved: {won_games}/{self.args['num_parallel_games']} ")

            won_games_total += won_matches
            games_played_total += self.args['num_parallel_games']
            win_rates.append(won_matches / self.args['num_selfPlay_iterations'])

            last_wind_rates.append(win_rates[-1])
            if len(last_wind_rates) > 5:
                last_wind_rates.pop(0)
            elapsed_time = time.time() - start_time
            print(f"Won games from {self.args['num_parallel_games']}: {won_matches}. Average {np.mean(last_wind_rates)}. {elapsed_time} seconds")
            print(win_rates)

            if np.mean(last_wind_rates) > best and len(last_wind_rates) > 3 and np.mean(last_wind_rates) > 0.4:
                best = np.mean(last_wind_rates)
                torch.save(value_model_raw.state_dict(), f'res/rl/value_simulation_{iteration + 25}.pth')
                torch.save(policy_model_raw.state_dict(), f'res/rl/policy_simulation_{iteration + 25}.pth')

            self.policy_model.train()
            self.value_model.train()
            self.train(memory)


        return win_rates

    def evaluate(self):
        won_games_total = 0
        games_played_total = 0
        hist = []
        self.policy_model.eval()
        self.value_model.eval()

        for iteration in range(self.args['num_iterations']):
            start = time.time()
            memory_new, won_games = self.selfPlay(eval_only=True)
            print(f"game solved: {won_games}/{self.args['num_parallel_games']} ")
            hist.append(won_games)
            won_games_total += won_games
            games_played_total += self.args['num_parallel_games']
            mean_games_won = won_games_total / games_played_total
            print(f"Won games from {self.args['num_parallel_games']}: {won_games}. Average {mean_games_won}")
            print(f"Time elapsed {time.time() - start}")
        return hist

class SPG:
    def __init__(self, game, validation_set = False):
        self.state = game.start_random_level_validate() if validation_set else game.start_random_level()
        self.memory = []
        self.root = None
        self.node = None
        self.visited_states = []

    def update_root(self, action):
        self.node = None
        new_root = None
        for node in self.root.children:
            if node.action_taken == action:
                new_root = node
        if new_root is None:
            return False
        else:
            self.root = new_root
            new_root.parent = None
            return True

##### TRAINING!
game = SokobanEnvWrapper()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model(images[:, :3, :, :], images[:, 3:, :, :])

policy_model_raw = EncoderResnetModel(5).cuda()
policy_model_raw.load_state_dict(torch.load(f'res/behavior_cloning/policy_model_synthetic.pth'))
policy_model = ModelWrapper(policy_model_raw)

value_model_raw = EncoderResnetModel(1, torch.nn.Sigmoid()).cuda()
value_model_raw.load_state_dict(torch.load(f'res/behavior_cloning/value_model_synthetic.pth'))
value_model = ModelWrapper(value_model_raw)
policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=3e-7)
value_optimizer = torch.optim.Adam(value_model.parameters(), lr=3e-7)

args = {
    'C': 1,
    'max_steps_per_level': 60,
    'num_searches': 75,
    'num_iterations': 180,
    'num_selfPlay_iterations': 20,
    'num_parallel_games': 20,
    'num_epochs': 1,
    'batch_size': 32,
    'dirichlet_epsilon': 0.05,
    'dirichlet_alpha': 2.0
}

alphaZero = AlphaZeroParallel(policy_model, policy_optimizer, value_model, value_optimizer, game, args)
win_rates = alphaZero.learn()
