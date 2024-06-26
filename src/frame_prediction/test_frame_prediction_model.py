# tests the frame prediction model

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.frame_prediction.frame_prediction_model import FramePredictionModel
from src.frame_prediction.imageds import ImageDS

batch_size = 8
val_batch_size = 32
model = FramePredictionModel(3, 5)
model.load_state_dict(torch.load(f'res/frame_prediction/frame_prediction_skin2.pth'))
model.cuda()
model.eval()

ds = ImageDS(skip_steps=0)
ds_iter = iter(ds)

start_frame, current_frame, next_frame, actions = next(ds_iter)
print(actions.shape)
predicted_next_frame = model(start_frame.unsqueeze(0).cuda(), current_frame.unsqueeze(0).cuda(), actions)
print(actions)
next_img_assumed = (np.moveaxis(predicted_next_frame[0].detach().cpu().numpy(), 0, -1) + 0.5) * 255
cv2.imwrite(f"tmp/predicted_change.png", cv2.cvtColor(next_img_assumed, cv2.COLOR_RGB2BGR))

next_img_assumed = (np.moveaxis(predicted_next_frame[0].detach().cpu().numpy() + current_frame.detach().cpu().numpy(), 0, -1) + 0.5) * 255
cv2.imwrite(f"tmp/predicted_next_frame.png", cv2.cvtColor(next_img_assumed, cv2.COLOR_RGB2BGR))

current_img = (np.moveaxis(current_frame.detach().cpu().numpy(), 0, -1) + 0.5) * 255
cv2.imwrite(f"tmp/current_frame.png", cv2.cvtColor(current_img, cv2.COLOR_RGB2BGR))

next_frame_img = (np.moveaxis(next_frame.detach().cpu().numpy(), 0, -1) + 0.5) * 255
cv2.imwrite(f"tmp/next_frame.png", cv2.cvtColor(next_frame_img, cv2.COLOR_RGB2BGR))

start_frame_img = (np.moveaxis(start_frame.detach().cpu().numpy(), 0, -1) + 0.5) * 255
cv2.imwrite(f"tmp/start_frame.png", cv2.cvtColor(start_frame_img, cv2.COLOR_RGB2BGR))

ds = ImageDS(skip_steps=40)
dataloader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
ds_iterator = iter(dataloader)
start_frame, current_frame, next_frame, actions = next(ds_iterator)

current_img = (np.moveaxis(current_frame[0].detach().cpu().numpy(), 0, -1) + 0.5) * 255
cv2.imwrite(f"tmp/current_frame.png", cv2.cvtColor(current_img, cv2.COLOR_RGB2BGR))

next_frame_img = (np.moveaxis(next_frame[0].detach().cpu().numpy(), 0, -1) + 0.5) * 255
cv2.imwrite(f"tmp/next_frame.png", cv2.cvtColor(next_frame_img, cv2.COLOR_RGB2BGR))

predicted_next_frame = model(start_frame.cuda(), current_frame.cuda(), actions[:, 0, :])[0]
next_img_assumed = (np.moveaxis(predicted_next_frame[0].detach().cpu().numpy(), 0, -1) + 0.5) * 255
cv2.imwrite(f"tmp/predicted_next_frame.png", cv2.cvtColor(next_img_assumed, cv2.COLOR_RGB2BGR))


for idx in range(actions.shape[1]):
    predicted_next_frame = model(start_frame.cuda(), current_frame.cuda(), actions[:, idx, :].squeeze())
    next_img_assumed = (np.moveaxis(predicted_next_frame[0].detach().cpu().numpy() + current_frame[0].detach().cpu().numpy(), 0, -1) + 0.5) * 255
    predicted_change = (np.moveaxis(predicted_next_frame[0].detach().cpu().numpy(), 0, -1) + 0.5) * 255
    cv2.imwrite(f"tmp/predicted_next_frame_{idx}.png", cv2.cvtColor(next_img_assumed, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"tmp/predicted_change_{idx}.png", cv2.cvtColor(predicted_change, cv2.COLOR_RGB2BGR))
    current_frame = current_frame + predicted_next_frame