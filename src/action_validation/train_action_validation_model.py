# Trains a model to be able to recognize implausible/invalid moves.

import glob
import random

import torch
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from src.action_validation.action_validatoin_model import ActionValidationModel
from src.common.abstract_image_ds import AbstractImageDS
from src.common.actions import ACTIONS
from src.common.image_folders import synthetic_training_folders
from src.common.preprocess_image import preprocess
from src.frame_prediction.frame_prediction_model import FramePredictionModel

num_actions = 5
# we use the FramePredictionModel to create proposed next states of the game
env_model = FramePredictionModel(3, num_actions)
env_model.load_state_dict(torch.load(f'res/frame_prediction/frame_prediction_model.pth'))
env_model.cuda()
env_model.eval()


class ImageDS(AbstractImageDS):
    def __init__(self, folders=None):
        super().__init__(folders)

    def __getitem__(self, idx):
        if idx == len(self.files) -1:
            idx -= 1
        file = self.files[idx]

        next_file = self.files[idx + 1]
        prefix = self.get_prefix(file)
        if not self.is_same_level(file, next_file):
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        f = open(f"{prefix}.txt".replace('_step_', ''), "r")
        solution = f.readline().split()
        f.close()
        moves_in_level = sorted(glob.glob(f"{prefix}*.png"))

        move = moves_in_level.index(file)

        action_letter = solution[move]
        if action_letter == 'z': # skip 'sleep action'
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        current_frame = Image.open(file)
        start_frame = Image.open(f"{prefix}00000.png")

        input_tensor1 = preprocess(start_frame)
        input_tensor2 = preprocess(current_frame)

        action = ACTIONS[action_letter]
        choose_correct_action = random.randint(0,1) == 1
        if choose_correct_action:
            action = torch.nn.functional.one_hot(torch.tensor(action), num_classes=num_actions).cuda()
        else:
            remaining_actions = [x for x in list(range(0, 4)) if x != action]
            random_action = random.choice(remaining_actions)
            action = torch.nn.functional.one_hot(torch.tensor(random_action), num_classes=num_actions).cuda()

        return input_tensor1, input_tensor2, action, choose_correct_action


def calculate_accuracy(pred, actual, cut_off = 0.5):
    predicted = (torch.nn.Sigmoid()(pred) > cut_off).float()
    return (predicted == actual).float().mean()


batch_size = 16
val_batch_size = 16

model = ActionValidationModel()
model.cuda()

loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100]).cuda())
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9998)

ds = ImageDS()
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

test_ds = ImageDS(synthetic_training_folders)
test_dataloader = DataLoader(test_ds, batch_size=val_batch_size, shuffle=True, num_workers=0)


train_losses = []
test_losses = []
test_accuracies = []
train_accuracies = []


print(f'Step | Train loss | Train Acc | Test loss | Accuracy')
mean_loss = 0
for step in range(0, 15000):
    ds_iterator = iter(dataloader)
    start_frame, current_frame, actions, is_correct_action = next(ds_iterator)
    predicted_image_change = env_model(start_frame, current_frame, actions)
    if random.randint(0, 1) == 1:
        start_frame = torch.flip(start_frame, [3])
        current_frame = torch.flip(current_frame, [3])
        predicted_image_change = torch.flip(predicted_image_change, [3])
    rotation = random.randint(0, 3)
    if rotation > 0:
        start_frame = torch.rot90(start_frame, k=rotation, dims=(2, 3))
        current_frame = torch.rot90(current_frame, k=rotation, dims=(2, 3))
        predicted_image_change = torch.rot90(predicted_image_change, k=rotation, dims=(2, 3))

    model.train()

    plausibility_prediction = model(current_frame, current_frame + predicted_image_change)

    loss = loss_fn(plausibility_prediction, is_correct_action.view(-1,1).float().cuda())

    mean_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if optimizer.param_groups[0]['lr'] > 1e-6:
        scheduler.step()

    if step % 500 == 0 and step != 0:
        if step > 0:
            mean_loss /= 500
        with torch.no_grad():
            test_loss = 0
            test_acc = 0
            train_acc = 0
            rounds = 10
            for _ in range(rounds):
                start_frame, current_frame, actions, is_correct_action = next(ds_iterator)
                predicted_image_change = env_model(start_frame, current_frame, actions)

                if random.randint(0, 1) == 1:
                    start_frame = torch.flip(start_frame, [3])
                    current_frame = torch.flip(current_frame, [3])
                    predicted_image_change = torch.flip(predicted_image_change, [3])
                rotation = random.randint(0, 3)
                if rotation > 0:
                    start_frame = torch.rot90(start_frame, k=rotation, dims=(2, 3))
                    current_frame = torch.rot90(current_frame, k=rotation, dims=(2, 3))
                    predicted_image_change = torch.rot90(predicted_image_change, k=rotation, dims=(2, 3))

                plausibility_prediction = model(current_frame, predicted_image_change + current_frame)
                train_acc += calculate_accuracy(plausibility_prediction, is_correct_action.view(-1,1).float().cuda())

                test_ds_iterator = iter(test_dataloader)
                start_frame, current_frame, actions, is_correct_action = next(ds_iterator)
                predicted_image_change = env_model(start_frame, current_frame, actions)

                if random.randint(0, 1) == 1:
                    start_frame = torch.flip(start_frame, [3])
                    current_frame = torch.flip(current_frame, [3])
                    predicted_image_change = torch.flip(predicted_image_change, [3])
                rotation = random.randint(0, 3)
                if rotation > 0:
                    start_frame = torch.rot90(start_frame, k=rotation, dims=(2, 3))
                    current_frame = torch.rot90(current_frame, k=rotation, dims=(2, 3))
                    predicted_image_change = torch.rot90(predicted_image_change, k=rotation, dims=(2, 3))

                plausibility_prediction = model(current_frame, predicted_image_change + current_frame)
                test_loss += loss_fn(plausibility_prediction, is_correct_action.view(-1,1).float().cuda())
                test_acc += calculate_accuracy(plausibility_prediction, is_correct_action.view(-1,1).float().cuda())

        print(f'{step:5d} | ' f'{mean_loss:.8f} | {train_acc/rounds:.2f} | {test_loss/rounds:.8f} | {test_acc/rounds:.2f}')

        train_losses.append(mean_loss)
        test_losses.append(test_loss/rounds)
        test_accuracies.append(test_acc/rounds)
        train_accuracies.append(train_acc/rounds)

        if step > 0 and step % 500 == 0:
            torch.save(model.state_dict(), f'res/action_validator/action_validator_{step}.pth')

        file = open("res/action_validator/train_losses.pt", "w")
        file.write(repr(train_losses))
        file.close()
        file = open("res/action_validator/test_losses.pt", "w")
        file.write(repr(test_losses))
        file.close()
        file = open("res/action_validator/test_accuracies.pt", "w")
        file.write(repr(test_accuracies))
        file.close()
        file = open("res/action_validator/train_accuracies.pt", "w")
        file.write(repr(train_accuracies))
        file.close()

        mean_loss = 0
