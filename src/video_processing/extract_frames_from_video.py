# Creates video frames videos in a folder, removes duplicate images and resizes the images to 224x244

import os
import shutil
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import glob
import subprocess

folder = "IN_FOLDER"
out_folder = "OUT_FOLDER"

model = models.resnet50(pretrained=True)
model.eval()

model = torch.nn.Sequential(*(list(model.children())[:-1]))

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

video_files = sorted(glob.glob(f'{folder}/*.mp4'))

current_level = 0
for in_file in video_files:

    in_file = os.path.basename(in_file)
    filename_without_extension = in_file.split(".")[0]
    t = filename_without_extension.replace('_', '')
    file_prefix = f"{t}_"
    command = f"ffmpeg -i \"{in_file}\" -vf \"scale=450:360\" \"{file_prefix}%05d.png\""
    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=folder)
    print("Output:", result.stdout)

    current_idx = 1
    formatted_idx = "{:05d}".format(current_idx)
    shutil.copy(f"{folder}{file_prefix}{formatted_idx}.png", f"{out_folder}{file_prefix}00000.png")
    counter = 1
    next_idx = 2
    very_first_image = Image.open(f"{folder}{file_prefix}{formatted_idx}.png")
    features_very_first_image = model(preprocess(very_first_image).unsqueeze(0))
    while True:
        formatted_idx = "{:05d}".format(current_idx)

        if not os.path.exists(f"{folder}{file_prefix}{formatted_idx}.png"):
            break

        image = Image.open(f"{folder}{file_prefix}{formatted_idx}.png")

        formatted_idx = "{:05d}".format(next_idx)

        if not os.path.exists(f"{folder}{file_prefix}{formatted_idx}.png"):
            break

        next_image = Image.open(f"{folder}{file_prefix}{formatted_idx}.png")

        input1 = preprocess(image).unsqueeze(0)  # Add batch dimension
        input2 = preprocess(next_image).unsqueeze(0)

        with torch.no_grad():
            features1 = model(input1)
            features2 = model(input2)

        distance = torch.norm(features1 - features2)

        if distance > 10:
            print(f"distance to large. Assuming next level started {current_idx}")
            break

        if distance < 0.25:
            current_idx += 1
            next_idx += 1
        else:
            print(f"{current_idx} and {next_idx} are not identical")
            formatted_idx = "{:05d}".format(next_idx)
            formatted_counter = "{:05d}".format(counter)
            shutil.copy(f"{folder}{file_prefix}{formatted_idx}.png", f"{out_folder}{file_prefix}{formatted_counter}.png")
            counter += 1
            current_idx = next_idx
            next_idx += 1

    command = f"rm *.png"
    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=folder)

    print(f"copied relevant frames. last file was {in_file}")


#####################################
# convert all files to jpg
#####################################

command = f"mogrify -format jpg *.png"
result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=out_folder)

command = f"rm *.png"
result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=out_folder)


#####################################
# resize images to end up with square shaped images
#####################################

images = sorted(glob.glob(f'{out_folder}/*.jpg'))

def make_square(file):
    image = Image.open(file)
    if image.size[0] != image.size[1]:
        print(f"file {file}")
        larger_side = max(image.size[0], image.size[1])
        command = f"convert \"{file}\" -gravity center -background black -extent {larger_side}x{larger_side} \"{file}\""
        subprocess.run(command, shell=True, capture_output=True, text=True)


for image in images:
    make_square(image)
print("All images are square shaped now.")


#####################################
# rename first file of each level to 00000
#####################################

def move_no_pos(file_name):
    return file_name.rindex('_') + 1


def get_prefix(file_name):
    return file_name[:move_no_pos(file_name)]


images = sorted(glob.glob(f'{out_folder}/*.jpg'))
old_prefix = None
for file in images:
    prefix = get_prefix(file)
    if old_prefix is None or old_prefix != prefix:
        old_prefix = prefix
        if file != f"{prefix}00000.jpg":
            pass
        print(f"{file} -> {prefix}00000.jpg")
print("done.")