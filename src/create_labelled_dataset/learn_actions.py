import random

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from src.common.image_folders import synthetic_training_folders
from src.create_labelled_dataset.unet import UNet
import glob

class ImageDS(Dataset):
    def __init__(self, folder = synthetic_training_folders[0], extension = 'png'):
        self.files = []
        self.extension = extension
        self.files = sorted(glob.glob(f'{folder}/*.{extension}'))
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
        if idx == len(self.files) -1:
            idx -= 1
        file = self.files[idx]
        next_file = self.files[idx+1]

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
            transforms.Resize((160, 160)),
            transforms.ToTensor()
        ])
        input_tensor0 = preprocess(start_frame)
        input_tensor1 = preprocess(current_frame)
        input_tensor2 = preprocess(next_frame)
        input_tensor0 -= 0.5
        input_tensor1 -= 0.5
        input_tensor2 -= 0.5

        rotations = random.randint(0, 3)
        for k in range(rotations):
            input_tensor0 = torch.rot90(input_tensor0, k=1, dims=(1, 2))
            input_tensor1 = torch.rot90(input_tensor1, k=1, dims=(1, 2))
            input_tensor2 = torch.rot90(input_tensor2, k=1, dims=(1, 2))
        if random.randint(0, 1) == 1:
            input_tensor0 = torch.flip(input_tensor0, dims=[2])
            input_tensor1 = torch.flip(input_tensor1, dims=[2])
            input_tensor2 = torch.flip(input_tensor2, dims=[2])

        return input_tensor0, input_tensor1, input_tensor2


action_no = 5 #
WAIT_ACTION = 4
unet = UNet(3, action_no).cuda()

action_count = np.zeros(action_no)
criterion = torch.nn.MSELoss()
action_count_penalty = 0.02
action_penalty_decay = 0.995
batch_size = 16

dataset = ImageDS()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
reconstruction_optimizer = torch.optim.Adam(unet.parameters(), lr=1e-5)

def find_best_action(frame_0, frame_1, frame_2, unet, batch_size, already_taken = None):
    global action_count
    global action_count_penalty
    global action_penalty_decay
    action_beta_decay = 0.95
    with torch.no_grad():
        unet.eval()
        losses = torch.zeros((batch_size, action_no))
        for action in range(0, action_no):
            one_hot_action = torch.nn.functional.one_hot(torch.tensor(action), num_classes=action_no).unsqueeze(0).expand(batch_size, action_no).cuda()
            predicted_output_image = unet(frame_0, frame_1, one_hot_action)
            loss = torch.nn.MSELoss(reduction='none')(predicted_output_image, frame_2 - frame_1)
            losses[:, action] = torch.mean(loss, dim=(1,2,3)) + action_count[action] * action_count_penalty
    if already_taken is not None and already_taken.size()[1] > 0:
        losses.scatter_(1, torch.ones((losses.shape[0], 1), dtype=torch.int64) * WAIT_ACTION, 10000) # discourage wait action if another action has already taken place
        losses.scatter_(1, torch.abs(already_taken), 10000 * ((already_taken + 1) / torch.abs(already_taken + 1))) # once taken WAIT will always be taken, others won't be taken again
    best_actions = torch.min(losses, dim=1)
    action_count *= action_beta_decay
    action_count_penalty *= action_penalty_decay
    action_count[best_actions.indices] += 1
    return torch.nn.functional.one_hot(best_actions.indices, num_classes=action_no).cuda(), best_actions.indices

train_losses = []
test_losses = []
print(f'Step | Train loss')
mean_loss = 0
for step in range(0, 40001):
    frame_0, frame_1, frame_2 = next(iter(dataloader))
    frame_0, frame_1, frame_2 = frame_0.type(torch.float32).cuda(), frame_1.type(torch.float32).cuda(), frame_2.type(torch.float32).cuda()
    unet.train()
    already_taken = torch.zeros(batch_size, 0, dtype=torch.int64)
    loss = None
    for k in range(0, 4):
        if k > 0:
            frame_1 = torch.rot90(frame_1, k=1, dims=(2, 3))
            frame_2 = torch.rot90(frame_2, k=1, dims=(2, 3))
        one_hot, action_indices = find_best_action(frame_0, frame_1, frame_2, unet, batch_size, already_taken)
        already_taken = torch.cat((already_taken, action_indices.unsqueeze(-1)), dim=1)
        if k == 0:
            # if WAIT_ACTION is the first action chosen, then WAIT_ACTION should always be taken no matter the orientation
            mask = already_taken == WAIT_ACTION
            already_taken[mask] *= -1
        predicted_image = unet(frame_0, frame_1, one_hot)
        if loss is None:
            loss = criterion(predicted_image, frame_2 - frame_1)
        else:
            loss += criterion(predicted_image, frame_2 - frame_1)

    reconstruction_optimizer.zero_grad()
    loss.backward()
    reconstruction_optimizer.step()
    mean_loss += loss.item() / 4

    if step % 200 == 0:
        if step > 199:
            mean_loss /= 200
        with torch.no_grad():
            print(action_count)
            train_losses.append(mean_loss)
            print(f'{step:5d} | {mean_loss:.8f}')
            mean_loss = 0


# manual action mapping
with torch.no_grad():
    unet.eval()
    frames_0, frames_1, frames_2 = next(iter(dataloader))
    frames_0, frames_1, frames_2 = frames_0.type(torch.float32).cuda(), frames_1.type(torch.float32).cuda(), frames_2.type(torch.float32).cuda()
    actions, _ = find_best_action(frames_0, frames_1, frames_2, unet, batch_size)

    predicted_changes = unet(frames_0, frames_1, actions)

    action_values = torch.max(actions, dim=1)
    print(action_values.indices)
    for idx in range(0, batch_size):
        frame_1 = (np.moveaxis(frames_1[idx].cpu().numpy(), 0, -1) + 0.5) * 255
        cv2.imwrite(f"tmp/{idx}_in_01.png", cv2.cvtColor(frame_1, cv2.COLOR_RGB2BGR))

        frame_2 = (np.moveaxis(frames_2[idx].cpu().numpy(), 0, -1) + 0.5) * 255
        cv2.imwrite(f"tmp/{idx}_in_02.png", cv2.cvtColor(frame_2, cv2.COLOR_RGB2BGR))

        image = (np.moveaxis(frames_1[idx].cpu().numpy() + predicted_changes[idx].cpu().numpy(), 0, -1) + 0.5) * 255
        cv2.imwrite(f"tmp/{idx}_prediction.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# manual action mapping
with torch.no_grad():
    unet.eval()
    frames_0, frames_1, frames_2 = next(iter(dataloader))
    frames_0, frames_1, frames_2 = frames_0.type(torch.float32).cuda(), frames_1.type(torch.float32).cuda(), frames_2.type(torch.float32).cuda()

    best_action, _ = find_best_action(frames_0, frames_1, frames_2, unet, batch_size)
    print(torch.argmax(best_action[0, :]))
    for action in range(action_no):
        one_hot_action = torch.nn.functional.one_hot(torch.tensor(action), num_classes=action_no).unsqueeze(0).expand(
            batch_size, action_no).cuda()
        predicted_changes = unet(frames_0, frames_1, one_hot_action)
        image = (np.moveaxis(frames_1[0].cpu().numpy() + predicted_changes[0].cpu().numpy(), 0, -1) + 0.5) * 255
        cv2.imwrite(f"tmp/{0}_prediction_{action}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    frame_1 = (np.moveaxis(frames_1[0].cpu().numpy(), 0, -1) + 0.5) * 255
    cv2.imwrite(f"tmp/{0}_in_01.png", cv2.cvtColor(frame_1, cv2.COLOR_RGB2BGR))

    frame_2 = (np.moveaxis(frames_2[0].cpu().numpy(), 0, -1) + 0.5) * 255
    cv2.imwrite(f"tmp/{0}_in_02.png", cv2.cvtColor(frame_2, cv2.COLOR_RGB2BGR))



