# General information 

This repository was used to extract actions from unlabeled video frames from Sokoban and subsequently
train a policy and value model for Sokoban on state-action pairs. 
It uses pytorch as DNN framework.
Furthermore, Reinforcement learning with Monte Carlo tree search is performed.

This repository contains code to train models to simulate Sokoban and determine which moves are valid.

# Installing libraries

Type 
pip install -m requirements.txt

# Stitching models together

Due to file size limiations on github some model files had to be split. They can be
stitched back together like so:

`
cat model_file_name.?? > model_file_name.pth
`

To recover all files simply execute `bash ./stitch.sh`

# Restrictions

This repo only contains few training and validation files such that you
can test if the training procedure works. With such few files you will
probably run into overfitting.

# Folder organisation

There are multiple sub folders under `src` performing different tasks.
Typically a folder contains a file to train a model (`train*` ) and a file
for testing a model (`test*`). In addition there are model files or files for datesets. 

I will briefly explain the purpose and content of each folder.

## sokoban_env

This folder contains a file to provide a Sokoban environment. 
`sokoban_env.py` was modified from https://github.com/mpSchrader/gym-sokoban

## create_labelled_dataset

This folder is used to extract actions form an unlabeled data set.
Execute `learn_actions.py` to start training.

## behavior_cloning

Train a policy and value model using frame-action pairs.
A small dataset is already available under `sokoban_solutions`.
Execute `train_policy_and_value_model.py` to start training.

## reinforcement learning

Perform reinforcement learning with a MCTS in a model-based or model-free manner.
Execute `rl_model_based.py` or `rl_model_free.py` to do so.
For model-based learning you must execute `bash ./stitch.sh` first.
The code was modified from https://github.com/foersterrobert/AlphaZeroFromScratch

## frame_prediction

Trains a model to predict the next frame given the current state of the game and an
action. To start training run the file `train_frame_prediction_model.py`

## action_validation

Trains a model to validate Sokoban moves.
Run `train_action_validation_model.py`. You must run `bash ./stitch.sh` first.

## game_state_recognition

Trains a model to determine if a level is finished or not.
Run `train_model.py` to start training.

# Starting the Sokoban simulator GUI

A prerequisite is that you executed `bash ./stitch.sh`.
Enter the src/sokoban_simulator folder and run `gui.py`

