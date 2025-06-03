import os


def turn_on_screen():
    """
    Turns on the Raspberry Pi screen with xset dpms force on command.
    """
    os.system("xset dpms force on")


def turn_off_screen():
    """
    Turns off the Raspberry Pi screen with xset dpms force off command.
    """
    os.system("xset dpms force off")


def set_screen_cooldown(seconds):
    """
    Sets the Raspberry Pi screen to turn off after a specified number of seconds.

    :param seconds: Number of seconds before the screen turns off.
    """
    os.system(f"xset dpms {seconds} {seconds} {seconds}")
