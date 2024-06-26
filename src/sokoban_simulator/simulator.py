import torch

from src.action_validation.action_validatoin_model import ActionValidationModel
from src.frame_prediction.frame_prediction_model import FramePredictionModel
from src.game_state_recognition.game_state_model import GameStateModel

action_no = 5

class Simulator:
    def __init__(self):
        self.game_state_model = GameStateModel()
        self.game_state_model.load_state_dict(torch.load('res/game_state_model/game_state_model.pth'))
        self.game_state_model.cuda()
        self.game_state_model.eval()
        self.move_validator = ActionValidationModel()
        self.move_validator.load_state_dict(torch.load('res/action_validator/action_validation_model.pth'))
        self.move_validator.cuda()
        self.move_validator.eval()
        self.frame_prediction_model = FramePredictionModel(3, 5)
        self.frame_prediction_model.load_state_dict(torch.load('res/frame_prediction/frame_prediction_model.pth'))
        self.frame_prediction_model.cuda()
        self.frame_prediction_model.eval()

    def map_letter_to_action(self, action):
        if action == 'l':
            return 0
        elif action == 'u':
            return 1
        elif action == 'r':
            return 2
        elif action == 'd':
            return 3
        raise f"Action {action} not mapped"

    def step_letter(self, first_frame, current_frame, action_letter):
        action = self.map_letter_to_action(action_letter)
        return self.step(first_frame, current_frame, action)

    def step(self, first_frame, current_frame, action):
        one_hot_action = torch.nn.functional.one_hot(torch.tensor(int(action), device="cuda"), num_classes=action_no).unsqueeze(0)
        return self.step_batch(first_frame, current_frame, one_hot_action)

    def is_terminated(self, first_frame, current_frame):
        return self.game_state_model(first_frame, current_frame)


    def step_batch(self, first_frames, current_frames, actions):
        frame_change = self.frame_prediction_model(first_frames, current_frames, actions)
        next_frame = current_frames + frame_change
        is_valid_move = self.move_validator(current_frames, next_frame)
        is_final = self.game_state_model(first_frames, next_frame)
        return next_frame, is_valid_move, is_final
