import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.common.image_folders import synthetic_training_folders
from src.sokoban_simulator.simulator import Simulator

simulator = Simulator()

input_image_1 = Image.open(f"{synthetic_training_folders[0]}/game_00150_step_00000.png")

preprocess = transforms.Compose([
    transforms.ToTensor()
])
input_tensor1 = preprocess(input_image_1)
input_tensor1 -= 0.5
next_frame = input_tensor1.unsqueeze(0)

next_frame_img = (np.moveaxis(next_frame[0].detach().numpy(), 0, -1) + 0.5) * 255
cv2.imwrite(f"tmp/3.png", cv2.cvtColor(next_frame_img, cv2.COLOR_RGB2BGR))

counter = 0
while True:
    next_action = input()
    next_frame, valid, is_final = simulator.step(input_tensor1.unsqueeze(0), next_frame, next_action)

    next_frame_img = (np.moveaxis(next_frame[0].detach().numpy(), 0, -1) + 0.5) * 255
    cv2.imwrite(f"tmp/{counter}.png", cv2.cvtColor(next_frame_img, cv2.COLOR_RGB2BGR))

    print(torch.nn.Sigmoid()(valid))
    print(torch.nn.Sigmoid()(is_final))
    counter += 1
