# create DS with image, next_image, real_or_not_real_move

import random
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from src.common.image_folders import synthetic_validation_folders
from src.frame_prediction.frame_prediction_model import FramePredictionModel
from src.frame_prediction.imageds import ImageDS

n_actions = 5

batch_size = 16
val_batch_size = 32
model = FramePredictionModel(3, n_actions)
model.cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9998)

ds = ImageDS(skip_steps=1)
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

test_ds = ImageDS(folders=synthetic_validation_folders, skip_steps=0)
test_dataloader = DataLoader(test_ds, batch_size=val_batch_size, shuffle=True, num_workers=0)

train_losses = []
test_losses = []
print(f'Step | Train loss | Test loss')
mean_loss = 0
for step in range(0, 100000):
    ds_iterator = iter(dataloader)
    start_frame, current_frame, next_frame, actions = next(ds_iterator)

    model.train()

    for idx in range(actions.shape[1]):
        predicted_change = model(start_frame, current_frame, actions[:, idx, :].squeeze())
        current_frame = current_frame + predicted_change

    loss = criterion(current_frame, next_frame)

    mean_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if step % 100 == 0 and step > 10000:
        skips = random.randint(1, 5)
        ds = ImageDS(skip_steps=skips)
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        test_ds = ImageDS(folders=synthetic_validation_folders, skip_steps=0)
        test_dataloader = DataLoader(test_ds, batch_size=val_batch_size, shuffle=True, num_workers=0)

    if step % 500 == 0 and step != 0:
        if step > 0:
            mean_loss /= 500
        with torch.no_grad():
            test_ds_iterator = iter(test_dataloader)
            start_frame, current_frame, next_frame, actions = next(test_ds_iterator)
            for idx in range(actions.shape[1]):
                predicted_change = model(start_frame, current_frame, actions[:, idx, :].squeeze())
                current_frame = current_frame + predicted_change

            test_loss = criterion(current_frame, next_frame)
            print(f'{step:5d} | ' f'{mean_loss:.8f} | {test_loss.item():.8f}')

            train_losses.append(mean_loss)
            test_losses.append(test_loss.item())

        mean_loss = 0

