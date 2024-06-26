# Creates a labeled dataset given a list of Sokoban images

import glob
import subprocess

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from src.create_labelled_dataset.unet import UNet

class ImageActionDS(Dataset):
    def __init__(self, extension = 'png'):
        self.files = []
        self.extension = extension
        for folder in ['sokoban_solutions']:
            self.files += sorted(glob.glob(f'{folder}/*.{extension}'))
        print(f"Found {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def move_no_pos(self, file_name):
        return file_name.rindex('_') + 1

    def get_prefix(self, file_name):
        return file_name[:self.move_no_pos(file_name)]

    def is_same_level(self, file_name1, file_name2):
        return self.get_prefix(file_name1) == self.get_prefix(file_name2)

    def __getitem__(self, idx):
        if idx == len(self.files) - 1:
            idx -= 1
        file = self.files[idx]
        next_file = self.files[idx + 1]

        prefix = self.get_prefix(file)
        if self.is_same_level(file, next_file):
            current_frame = file
            next_frame = next_file
            start_frame = f"{prefix}00000.{self.extension}"
        else:
            current_frame = self.files[idx - 1]
            next_frame = file
            start_frame = f"{prefix}00000.{self.extension}"

        current_frame = Image.open(current_frame)
        next_frame = Image.open(next_frame)
        start_frame = Image.open(start_frame)

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        input_tensor0 = preprocess(start_frame)
        input_tensor1 = preprocess(current_frame)
        input_tensor2 = preprocess(next_frame)
        input_tensor0 -= 0.5
        input_tensor1 -= 0.5
        input_tensor2 -= 0.5

        # action = 'r' if solution[move_no].lower() == 'l' else 'l' if solution[move_no].lower() == 'r' else solution[move_no].lower()
        return input_tensor0, input_tensor1, input_tensor2, self.get_prefix(file)

def map_action_to_letter(action_no):
    if action_no == 0:
        return 'r'
    if action_no == 1:
        return 'u'
    if action_no == 2:
        return 'l'
    if action_no == 3:
        return 'd'
    if action_no == 4:
        return 'z'
    raise f"Action {action_no} not mapped"

action_no = 5
unet = UNet(3, action_no).cuda()
unet.eval()
unet.load_state_dict(torch.load(f"res/move_prediction/action_extraction_model.pth"))

batch_size = 32

dataset = ImageActionDS()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

def find_best_action(frame_0, frame_1, frame_2, batch_size):
    with torch.no_grad():
        losses = torch.zeros((batch_size, action_no)).cuda()
        for action in range(action_no):
            one_hot_action = torch.nn.functional.one_hot(torch.tensor(action), num_classes=action_no).unsqueeze(0).expand(batch_size, action_no).cuda()
            predicted_output_image = unet(frame_0, frame_1, one_hot_action)
            loss = torch.nn.MSELoss(reduction='none')(predicted_output_image, frame_2 - frame_1)
            losses[:, action] = torch.mean(loss, dim=(1,2,3))
    best_actions = torch.min(losses, dim=1)
    return best_actions.indices.cuda()

batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
def create_labelled_data():
    number_of_items = int(dataloader.__len__())
    data_loader_iterator = iter(dataloader)
    print(f"Steps: {number_of_items}")
    actions = ''
    old_prefix = None
    for _ in range(number_of_items):
        frame_0, frame_1, frame_2, file_prefix = next(data_loader_iterator)
        file_prefix = str(file_prefix[0])
        best_action = find_best_action(frame_0.cuda(), frame_1.cuda(), frame_2.cuda(), batch_size)
        for idx in range(batch_size):
            if old_prefix is None or old_prefix != file_prefix:
                if old_prefix is not None:
                    print(f"finished {old_prefix}. writing file...")
                    with open(f'{old_prefix}.extracted.txt', 'w') as f:
                        f.write(actions)
                print(f"looking a level {file_prefix}")
                old_prefix = file_prefix
                actions = ''
            if actions != '':
                actions += ' '
            actions += str(map_action_to_letter(best_action[idx].item()))
    if actions != '':
        with open(f'{old_prefix}.txt', 'w') as f:
            f.write(actions)

create_labelled_data()


### remove z-moves
def move_no_pos(file_name):
    return file_name.rindex('_') + 1

def get_prefix(file_name):
    return file_name[:move_no_pos(file_name)]

def frame_number(file_name):
    return int(file_name[move_no_pos(file_name):-4])

def frame_number_digits_count(file_name):
    return len(file_name[move_no_pos(file_name):-4])

folder = 'temp'

def remove_z_moves():
    to_delete = []
    old_prefix = None
    files = sorted(glob.glob(f"{folder}/*.jpg"))
    remove_next = False
    current_move = 0
    for file in files:
        current_move += 1
        if remove_next:
            to_delete.append(file)
            remove_next = False
            continue
        file_prefix = get_prefix(file)
        if old_prefix is None or old_prefix != file_prefix:
            old_prefix = file_prefix
            print(f"{file_prefix}.txt")
            f = open(f"{file_prefix}.txt", "r")
            solution = f.readline().split()
            f.close()
            current_move = 0
        if current_move < len(solution) and solution[current_move] == 'z':
            if current_move == 0:
                remove_next = True
                continue
            to_delete.append(file)
    for file in to_delete:
        command = f"rm \"{file}\""
        print(f"removing {file}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=folder)

remove_z_moves()