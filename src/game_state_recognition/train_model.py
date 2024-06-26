# Trains a model to recognize if a Sokoban level is finished or not

import glob
import random
import torch.nn
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from src.common.abstract_image_ds import AbstractImageDS
from src.common.image_folders import synthetic_validation_folders
from src.common.preprocess_image import preprocess
from src.game_state_recognition.game_state_model import GameStateModel


class ImageDS(AbstractImageDS):
    def __init__(self, folders=None):
        super().__init__(folders)

    def __getitem__(self, idx):
        if idx == len(self.files) -1:
            idx -= 1
        file = self.files[idx]

        prefix = self.get_prefix(file)

        frames_in_level = sorted(glob.glob(f"{prefix}*.png"))

        use_final_image = random.randint(0, 1)

        if use_final_image == 1 or frames_in_level.index(file) == frames_in_level[-1]:
            file = frames_in_level[-1]

        current_frame = Image.open(file)
        start_frame = Image.open(f"{prefix}00000.png")

        input_tensor1 = preprocess(start_frame)
        input_tensor2 = preprocess(current_frame)

        flip = random.randint(0, 1)
        if flip == 0:
            input_tensor1 = torch.flip(input_tensor1, [2])
            input_tensor2 = torch.flip(input_tensor2, [2])

        rot = random.randint(0, 3)
        if rot > 0:
            input_tensor1 = torch.rot90(input_tensor1, k = rot, dims=(1,2))
            input_tensor2 = torch.rot90(input_tensor2, k = rot, dims=(1,2))

        return input_tensor1, input_tensor2, use_final_image


def calculate_accuracy(pred, actual):
    predicted = (pred > 0.5).float()
    return (predicted == actual).float().mean()


batch_size = 16

model = GameStateModel()
model.cuda()

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9998)

ds = ImageDS()
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

test_ds = ImageDS(synthetic_validation_folders)
test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=0)

print(f'Step | Train loss | Train Acc | Test loss | Accuracy')
for step in range(100000):
    ds_iterator = iter(dataloader)
    start_image, current_image, is_final = next(ds_iterator)
    model.train()
    prediction = model(start_image.cuda(), current_image.cuda())
    loss = criterion(prediction, is_final.view(-1,1).float().cuda())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if step % 500 == 0:
        with torch.no_grad():
            model.eval()
            start_image, current_image, is_final = next(ds_iterator)
            prediction = model(start_image.cuda(), current_image.cuda())
            train_loss = criterion(prediction, is_final.view(-1,1).float().cuda())
            train_acc = calculate_accuracy(torch.nn.Sigmoid()(prediction), is_final.view(-1,1).float().cuda())

            test_ds_iterator = iter(test_dataloader)
            start_image, current_image, is_final = next(test_ds_iterator)
            prediction = model(start_image.cuda(), current_image.cuda())
            test_loss = criterion(prediction, is_final.view(-1,1).float().cuda())
            test_acc = calculate_accuracy(torch.nn.Sigmoid()(prediction), is_final.view(-1,1).float().cuda())
            print(f'{step:5d} | ' f'{train_loss.item():.8f} | {train_acc.item():.2f} | {test_loss.item():.8f} | {test_acc.item():.2f}')

            if test_acc == 1.0 and train_acc == 1.0:
                torch.save(model.state_dict(), f'res/game_state_model/game_state_model_{step}.pth')
