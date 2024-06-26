# A alternative render utils file to display Sokoban using different icons

import numpy as np
import imageio

assets_folder = '/sokoban_skins/alternative/'
asset_size = 16

# Load images, representing the corresponding situation
box_filename = assets_folder.join(('res', 'box.jpg'))
box = imageio.imread(box_filename)

box_on_target_filename = assets_folder.join(('res', 'box_on_target.jpg'))
box_on_target = imageio.imread(box_on_target_filename)

box_target_filename = assets_folder.join(('res', 'box_target.jpg'))
box_target = imageio.imread(box_target_filename)

floor_filename = assets_folder.join(('res', 'floor.jpg'))
floor = imageio.imread(floor_filename)

player_filename = assets_folder.join(('res', 'player.jpg'))
player = imageio.imread(player_filename)

player_on_target_filename = assets_folder.join(('res', 'player_on_target.jpg'))
player_on_target = imageio.imread(player_on_target_filename)

wall_filename = assets_folder.join(('res', 'wall.jpg'))
wall = imageio.imread(wall_filename)

def room_to_rgb_alt(room, room_structure=None):
    """
       Creates an RGB image of the room.
       :param room:
       :param room_structure:
       :return:
       """
    resource_package = __name__

    room = np.array(room)
    if not room_structure is None:
        # Change the ID of a player on a target
        room[(room == 5) & (room_structure == 2)] = 6

    surfaces = [wall, floor, box_target, box_on_target, box, player, player_on_target]

    def skip(room, i, j):
        wall = 0
        if room[i, j] != wall:
            return False
        if i > 0 and j > 0 and room[i - 1, j - 1] != wall:
            return False
        if i > 0 and room[i - 1, j] != wall:
            return False
        if j > 0 and room[i, j - 1] != wall:
            return False
        if i > 0 and j < room.shape[1] - 1 and room[i - 1, j + 1] != wall:
            return False
        if j > 0 and i < room.shape[0] - 1 and room[i + 1, j - 1] != wall:
            return False
        if i < room.shape[0] - 1 and room[i + 1, j] != wall:
            return False
        if j < room.shape[1] - 1 and room[i, j + 1] != wall:
            return False
        if j < room.shape[1] - 1 and i < room.shape[0] - 1  and room[i + 1, j + 1] != wall:
            return False
        return True

    # Assemble the new rgb_room, with all loaded images
    room_rgb = np.zeros(shape=(room.shape[0] * 16, room.shape[1] * 16, 3), dtype=np.uint8)
    for i in range(room.shape[0]):
        x_i = i * 16

        for j in range(room.shape[1]):
            if skip(room, i, j):
                continue
            y_j = j * 16
            surfaces_id = room[i, j]

            room_rgb[x_i:(x_i + 16), y_j:(y_j + 16), :] = surfaces[surfaces_id]

    return room_rgb

