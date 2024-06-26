'''
This code is adapted from
https://github.com/foersterrobert/AlphaZeroFromScratch
Credit goes to the original author
'''


import hashlib
import os
import time
import random
import math
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from torch import nn

from src.common.encoder_resnet_model import EncoderResnetModel

print(np.__version__)

import torch

print(torch.__version__)

torch.manual_seed(0)

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


class SokobanEnvWrapper:
    def __init__(self):
        self.env = gym.make('Sokoban-v2', render_mode="rgb_array", reset=False)
        self.action_size = 5
        self.level_counter = 0
        level_counter_formatted = "{:05d}".format(self.level_counter)
        while os.path.exists(f"sokoban_levels/level_{level_counter_formatted}.npy"):
            self.level_counter += 1
            level_counter_formatted = "{:05d}".format(self.level_counter)
        self.level_counter -= 1
        print(f"levels found: {self.level_counter}")
        self.validate_level = 4800

    def start_level_validate(self):
        level_counter_formatted = "{:05d}".format(self.validate_level)
        start_state = np.load(f"sokoban_levels_validate/level_{level_counter_formatted}.npy")
        state = np.copy(start_state)
        self.env.reset(options={'state': state, 'start_state': start_state })
        self.validate_level += 1
        return [start_state, state]

    def start_random_level(self, level_no = None):
        level_no = level_no if level_no is not None else random.randint(0, self.level_counter)
        level_counter_formatted = "{:05d}".format(level_no)
        start_state = np.load(f"sokoban_levels/level_{level_counter_formatted}.npy")
        state = np.load(f"sokoban_levels/level_{level_counter_formatted}.npy")
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

    MODEL_ACTIONS_TO_ENV_ACTIONS = {
        0: 3,
        1: 1,
        2: 4,
        3: 2,
        4: 0
    }

    def get_next_state(self, state, action):
        self.env.reset(options={ 'state': state[1], 'start_state': state[0] } )
        observation, reward, terminated, truncated, info = self.env.step(self.MODEL_ACTIONS_TO_ENV_ACTIONS[action])
        return [state[0], observation], info["action.moved_player"] or info["action.moved_box"]

    def get_valid_moves(self, state):
        moves = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        for idx in range(4):
            self.env.reset(options={ 'state': state[1], 'start_state': state[0] } )
            observation, reward, terminated, truncated, info = self.env.step(self.MODEL_ACTIONS_TO_ENV_ACTIONS[idx])
            moves[idx] = 1.0 if info["action.moved_player"] or info["action.moved_box"] else 0.0
        return moves

    def get_value_and_terminated(self, state):
        self.env.reset(options={ 'state': state[1], 'start_state': state[0] })
        _, reward, done, _, _ = self.env.info()
        return 1 if done and reward > 10 else 0, done

    def get_encoded_state(self, state):
        self.env.reset(options={'state': state[0]})
        initial_state_render = self.env.render()
        self.env.reset(options={ 'state': state[1], 'start_state': state[0] })
        current_render = self.env.render()
        images_np = np.concatenate((initial_state_render, current_render), axis=2)
        return images_np.transpose((2, 0, 1)) / 255.0 - 0.5

    def get_encoded_states(self, states):
        images_stacked = []
        for idx in range(states.shape[0]):
            images_stacked.append(self.get_encoded_state(states[idx]))
        return np.array(images_stacked)

    def render_state(self, state):
        self.env.reset(options={ 'state': state[1], 'start_state': state[0] })
        return self.env.render()


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
            q_value = (self.value_sum / self.visit_count) # parent value as estimate
        else:
            q_value = (child.value_sum / child.visit_count)
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self, policy, existing_state_hashes):
        new_states = []
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state, move_valid = self.game.get_next_state(child_state, action)
                state_hash = hashlib.sha256(child_state[1].tobytes()).hexdigest()
                if move_valid and state_hash not in existing_state_hashes:
                    child = Node(self.game, self.args, child_state, self, action, prob)
                    new_states.append(state_hash)
                    self.children.append(child)
        return new_states

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
            policy = self.policy_model(torch.tensor(self.game.get_encoded_states(states), dtype=torch.float32, device="cuda"))
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
                new_hashes = spg.root.expand(spg_policy, spg.visited_states)
                spg.visited_states += new_hashes

        for search in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(node.state)

                if is_terminal:
                    node.backpropagate(value * 10) # make sure to use path that terminates

                else:
                    spg.node = node

            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if
                                  spGames[mappingIdx].node is not None]

            if len(expandable_spGames) > 0:
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])

                policy = self.policy_model(
                    torch.tensor(self.game.get_encoded_states(states), dtype=torch.float32, device="cuda"))
                value = self.value_model(
                    torch.tensor(self.game.get_encoded_states(states), dtype=torch.float32, device="cuda"))
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

                new_states = node.expand(spg_policy, spGames[mappingIdx].visited_states)
                spGames[mappingIdx].visited_states += new_states

                if len(new_states) == 0:
                    node.backpropagate(-1)
                    node.remove_from_parent()
                else:
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

            for i in range(len(spGames))[::-1]:
                spg = spGames[i]

                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                valid_moves = self.game.get_valid_moves(spg.state)
                action_probs = action_probs * valid_moves
                if np.sum(action_probs) > 0:
                    action_probs /= np.sum(action_probs)
                else:
                    action_probs = valid_moves / np.sum(valid_moves)

                action = np.argmax(action_probs)

                one_hot_vector = np.zeros(self.game.action_size, dtype=int)
                one_hot_vector[action] = 1
                spg.memory.append((spg.root.state, one_hot_vector))

                spg.state, is_valid = self.game.get_next_state(spg.state, action)

                value, is_terminal = self.game.get_value_and_terminated(spg.state)

                root_update_success = spg.update_root(action)
                if is_terminal or steps > max_steps or not root_update_success:
                    if is_terminal and value > 0:
                        won_games += 1
                    for idx, (hist_neutral_state, selected_action) in enumerate(spg.memory):
                        gamma = 0.95
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            selected_action,
                            value * gamma ** (steps - idx)
                        ))
                    del spGames[i]

            steps += 1

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

            if np.mean(last_wind_rates) > best and len(last_wind_rates) > 4 and np.mean(last_wind_rates) > 0.4:
                best = np.mean(last_wind_rates)
                torch.save(value_model_raw.state_dict(), f'res/rl/random_value_skin2_{iteration}.pth')
                torch.save(policy_model_raw.state_dict(), f'res/rl/random_policy_skin2_{iteration}.pth')

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
        self.state = game.start_level_validate() if validation_set else game.start_random_level()
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
    'num_iterations': 130,
    'num_selfPlay_iterations': 20,
    'num_parallel_games': 20,
    'num_epochs': 1,
    'batch_size': 32,
    'dirichlet_epsilon': 0.05,
    'dirichlet_alpha': 2.0
}

alphaZero = AlphaZeroParallel(policy_model, policy_optimizer, value_model, value_optimizer, game, args)
win_rates = alphaZero.learn()

######################
###################### validate
######################
game = SokobanEnvWrapper()
args = {
    'C': 1,
    'max_steps_per_level': 60,
    'num_searches': 75,
    'num_iterations': 5,
    'num_selfPlay_iterations': 20,
    'num_parallel_games': 20,
    'num_epochs': 1,
    'batch_size': 32,
    'dirichlet_epsilon': 0.05,
    'dirichlet_alpha': 2.0
}

alphaZero = AlphaZeroParallel(policy_model, None, value_model, None, game, args)
hist = alphaZero.evaluate()
print(hist)