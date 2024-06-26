import glob
from torch.utils.data import Dataset

from src.common.image_folders import synthetic_training_folders


class AbstractImageDS(Dataset):
    """A class to load Sokoban images and the action to solve a level which are provided as text file
    The __getitem__ function needs to be implemented
    """
    def __init__(self, folders = None, extension = 'png'):
        self.folders = synthetic_training_folders if folders == None else folders
        self.solutions = []
        self.files = []
        for folder in self.folders:
            self.solutions += sorted(glob.glob(f'{folder}/*.txt'))
            self.files += sorted(glob.glob(f'{folder}/*.{extension}'))
        print(f"Found {len(self.files)} images in total")
        print(f"Found {len(self.solutions)} levels in total")

    def __len__(self):
        return len(self.files)

    def move_no_pos(self, file_name):
        return file_name.rindex('_') + 1

    def get_prefix(self, file_name):
        return file_name[:self.move_no_pos(file_name)]

    def is_same_level(self, file_name1, file_name2):
        return self.get_prefix(file_name1) == self.get_prefix(file_name2)

    def __getitem__(self, idx):
        pass
