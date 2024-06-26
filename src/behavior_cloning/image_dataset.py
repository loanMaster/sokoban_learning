import glob
import random

import torch
from PIL import Image
from torchvision import transforms

from src.common.abstract_image_ds import AbstractImageDS
from src.common.actions import ACTIONS

reward_for_solving_level = 1
gamma = 0.95

class ImageDS(AbstractImageDS):
    def __init__(self, folders=None, image_extension = 'png'):
        super().__init__(folders, image_extension)
        self.img_extension = image_extension

    def __getitem__(self, idx):
        if idx == len(self.files) -1:
            idx -= 1
        file = self.files[idx]
        next_file = self.files[idx+1]

        if not self.is_same_level(file, next_file):
            idx -= 1
            file = self.files[idx]
            next_file = self.files[idx + 1]

        prefix = self.get_prefix(file)
        f = open(f"{prefix}.txt".replace('_step_', ''), "r")
        solution = f.readline().split()
        f.close()

        moves_in_level = sorted(glob.glob(f"{prefix}*.{self.img_extension}"))
        move = moves_in_level.index(file)

        action_letter = solution[move]
        if action_letter == 'z': # skip 'sleep action'
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        action = ACTIONS[action_letter]

        current_frame = Image.open(file)
        start_frame = Image.open(f"{prefix}00000.{self.img_extension}")

        preprocess = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor()
        ])
        input_tensor1 = preprocess(start_frame).cuda()
        input_tensor2 = preprocess(current_frame).cuda()
        input_tensor1 -= 0.5
        input_tensor2 -= 0.5

        if random.randint(0, 1) == 0: # flip horizontally
            action = ACTIONS['l'] if action == ACTIONS['r'] else ACTIONS['r'] if action == ACTIONS['l'] else action
            input_tensor1 = torch.flip(input_tensor1, [2])
            input_tensor2 = torch.flip(input_tensor2, [2])
        if random.randint(0, 1) == 0: # flip vertically
            action = ACTIONS['u'] if action == ACTIONS['d'] else ACTIONS['d'] if action == ACTIONS['u'] else action
            input_tensor1 = torch.flip(input_tensor1, [1])
            input_tensor2 = torch.flip(input_tensor2, [1])
        if random.randint(0, 1) == 0:  # swap alongside diagonal
            if action == ACTIONS['r']:
                action = ACTIONS['d']
            elif action == ACTIONS['l']:
                action = ACTIONS['u']
            elif action == ACTIONS['d']:
                action = ACTIONS['r']
            elif action == ACTIONS['u']:
                action = ACTIONS['l']
            input_tensor1 = torch.transpose(input_tensor1, 1, 2)
            input_tensor2 = torch.transpose(input_tensor2, 1, 2)
        if random.randint( 0, 1) == 0: # rotate 90%
            action = ACTIONS['r'] if action == ACTIONS['d'] else ACTIONS['l'] if action == ACTIONS[
                'u'] else \
                ACTIONS['d'] if action == ACTIONS['l'] else ACTIONS['u']
            input_tensor1 = torch.rot90(input_tensor1, k = 1, dims=(1,2))
            input_tensor2 = torch.rot90(input_tensor2, k = 1, dims=(1,2))
        elif random.randint( 0, 1) == 0: # rotate -90%
            action = ACTIONS['l'] if action == ACTIONS['d'] else ACTIONS['r'] if action == ACTIONS['u'] else \
                 ACTIONS['u'] if action == ACTIONS['l'] else ACTIONS['d']
            input_tensor1 = torch.rot90(input_tensor1, k = 3, dims=(1,2))
            input_tensor2 = torch.rot90(input_tensor2, k = 3, dims=(1,2))

        steps_to_goal = len(solution) - move
        value = reward_for_solving_level * (gamma ** steps_to_goal)
        return input_tensor1, input_tensor2, input_tensor2, torch.tensor(action).cuda(), torch.tensor(value).cuda()

