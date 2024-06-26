def map_letter_to_action(action):
    """A Mapping of an action letter (left, up, right, down, sleep) to
    their respective number used in various models
    """
    if action == 'u':
        return 1
    elif action == 'r':
        return 2
    elif action == 'd':
        return 3
    elif action == 'l':
        return 0
    raise f"Action {action} not mapped"
