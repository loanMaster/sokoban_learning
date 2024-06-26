import tkinter as tk
import numpy as np
import torch
from PIL import Image, ImageTk
from torchvision import transforms

from src.common.image_folders import synthetic_training_folders
from src.sokoban_simulator.simulator import Simulator

simulator = Simulator()

input_image_1 = Image.open(f"{synthetic_training_folders[0]}/game_00150_step_00000.png")
input_image_2 = Image.open(f"{synthetic_training_folders[0]}/game_00150_step_00000.png")

preprocess = transforms.Compose([
    transforms.ToTensor()
])
input_tensor1 = preprocess(input_image_1)
input_tensor1 -= 0.5
input_tensor1.cuda()

current_frame = preprocess(input_image_2)
current_frame -= 0.5
current_frame = current_frame.unsqueeze(0).cuda()

def move(action):
    global current_frame
    move_valid_indicator.config(text="")
    game_state_indicator.config(text="")
    current_frame, valid, is_final = simulator.step(input_tensor1.unsqueeze(0).cuda(), current_frame.cuda(), action)
    current_frame = torch.clamp(current_frame, min=-0.5, max=0.5)
    image_frame = (np.moveaxis(current_frame[0].cpu().detach().numpy(), 0, -1) + 0.5) * 255
    recreate_canvas(image_frame.astype(np.uint8))

    print(torch.nn.Sigmoid()(valid))
    print(torch.nn.Sigmoid()(is_final))
    move_valid_indicator.config(text="Yes" if torch.nn.Sigmoid()(valid) > 0.5 else "No")
    game_state_indicator.config(text="Yes" if torch.nn.Sigmoid()(is_final) > 0.5 else "No")


def up():
    move(1)

def left():
    move(0)

def down():
    move(3)

def right():
    move(2)

root = tk.Tk()
root.geometry("300x400")
root.title("Sokoban DNN Simulator")

frame_image = tk.Frame(root, height=270)
frame_image.pack(pady=10)
frame_move_valid = tk.Frame(root)
frame_move_valid.pack(pady=10)
frame_indicators = tk.Frame(root)
frame_indicators.pack(pady=10)
frame_buttons = tk.Frame(root)
frame_buttons.pack(pady=10)

canvas = tk.Canvas(frame_image, width=300, height=180)
canvas.pack()

def recreate_canvas(image_array):
    global canvas, image_tk
    canvas.destroy()
    canvas = tk.Canvas(frame_image, width=300, height=180)
    canvas.pack()
    image_pil = Image.fromarray(image_array)
    image_tk = ImageTk.PhotoImage(image_pil)
    canvas.create_image(75, 20, anchor=tk.NW, image=image_tk)

image_frame = (np.moveaxis(current_frame[0].cpu().detach().numpy(), 0, -1) + 0.5) * 255
recreate_canvas(image_frame.astype(np.uint8))

move_valid_label = tk.Label(frame_move_valid, text="Move valid?")
move_valid_indicator = tk.Label(frame_move_valid, text="Yes")
move_valid_label.pack(side=tk.LEFT)
move_valid_indicator.pack(side=tk.LEFT)

game_state_label = tk.Label(frame_indicators, text="Game finished?")
game_state_indicator = tk.Label(frame_indicators, text="No")
game_state_label.pack(side=tk.LEFT)
game_state_indicator.pack(side=tk.LEFT)

btn_up = tk.Button(frame_buttons, text="Up", command=up)
btn_left = tk.Button(frame_buttons, text="Left", command=left)
btn_down = tk.Button(frame_buttons, text="Down", command=down)
btn_right = tk.Button(frame_buttons, text="Right", command=right)

btn_up.grid(row=0, column=1)
btn_left.grid(row=1, column=0)
btn_down.grid(row=1, column=1)
btn_right.grid(row=1, column=2)

root.mainloop()


