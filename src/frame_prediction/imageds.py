import glob
import random
import torch
from PIL import Image
from src.common.abstract_image_ds import AbstractImageDS
from src.common.actions import ACTIONS
from src.common.preprocess_image import preprocess

class ImageDS(AbstractImageDS):
    def __init__(self, folders=None, skip_steps=0, original_ui = False):
        super().__init__(folders, extension='jpg' if original_ui else 'png')
        self.skip_steps = skip_steps
        self.extension = 'jpg' if original_ui else 'png'

    def __getitem__(self, idx):
        if idx == len(self.files) -1:
            idx -= 1
        file = self.files[idx]

        if idx+1+self.skip_steps >= self.__len__():
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        next_file = self.files[idx+1+self.skip_steps]

        prefix = self.get_prefix(file)
        if not self.is_same_level(file, next_file):
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        f = open(f"{prefix}.txt".replace('_step_', ''), "r") if self.extension == 'png' else open(f"{prefix}.txt", "r")
        solution = f.readline().split()
        f.close()
        moves_in_level = sorted(glob.glob(f"{prefix}*.{self.extension}"))

        move = moves_in_level.index(file)
        move_stop = moves_in_level.index(next_file)

        action_sequence = []
        offset = 0
        while move + offset < move_stop:
            action_letter = solution[move + offset]
            if action_letter == 'z': # skip 'sleep action'
                return self.__getitem__(random.randint(0, self.__len__() - 1))
            action_sequence.append(ACTIONS[action_letter])
            offset += 1

        current_frame = Image.open(file)
        next_frame = Image.open(next_file)
        start_frame = Image.open(f"{prefix}00000.{self.extension}")

        input_tensor1 = preprocess(start_frame)
        input_tensor2 = preprocess(current_frame)
        input_tensor3 = preprocess(next_frame)

        flip = random.randint(0, 1)
        if flip == 0: # flip horizontally
            for idx, action in enumerate(action_sequence):
                action_sequence[idx] = ACTIONS['l'] if action == ACTIONS['r'] else ACTIONS['r'] if action == ACTIONS['l'] else action
            input_tensor1 = torch.flip(input_tensor1, [2])
            input_tensor2 = torch.flip(input_tensor2, [2])
            input_tensor3 = torch.flip(input_tensor3, [2])

        return input_tensor1, input_tensor2, input_tensor3, torch.nn.functional.one_hot(torch.tensor(action_sequence), num_classes=5).cuda()
