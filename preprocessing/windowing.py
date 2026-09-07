"""
Windowing utilities.

DAGHAR standardized_view is already windowed into
3-second non-overlapping windows sampled at 20 Hz.

Therefore each sample already contains 60 timesteps.

This module will be used later when raw sensor recordings
or additional datasets are processed.
"""


WINDOW_SECONDS = 3
SAMPLING_RATE = 20

WINDOW_SIZE = WINDOW_SECONDS * SAMPLING_RATE


def get_window_size():
    return WINDOW_SIZE