"""Shared movement bindings for the viewer, terminal and X11 key polling."""

MOTION_COMMANDS = {
    "8": (.7, 0., 0.),
    "5": (-.7, 0., 0.),
    "4": (0., .5, 0.),
    "6": (0., -.5, 0.),
    "7": (0., 0., .6),
    "9": (0., 0., -.6),
}


def glfw_motion_key(keycode):
    """Accept both the top number row and GLFW's numeric keypad keys."""
    if 48 <= keycode <= 57:
        key = chr(keycode)
    elif 320 <= keycode <= 329:
        key = str(keycode - 320)
    else:
        return None
    return key if key in MOTION_COMMANDS else None
