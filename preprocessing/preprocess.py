import numpy as np
import pandas as pd

from config import SENSOR_CHANNELS, WINDOW_SIZE, LABEL_COLUMN


def get_sensor_columns(
    dataframe: pd.DataFrame,
    sensor_name: str,
):
    """
    Find and correctly order columns for one sensor axis.

    Example:
        accel-x-0
        accel-x-1
        ...
        accel-x-59
    """

    columns = [
        column
        for column in dataframe.columns
        if column.startswith(sensor_name + "-")
    ]

    def get_timestep(column_name):
        return int(column_name.rsplit("-", 1)[1])

    columns = sorted(
        columns,
        key=get_timestep,
    )

    return columns


def dataframe_to_numpy(dataframe: pd.DataFrame):
    """
    Convert DAGHAR dataframe into:

        X -> [samples, channels, timesteps]
        y -> [samples]

    Expected X shape:
        (N, 6, 60)
    """

    sensor_arrays = []

    for sensor in SENSOR_CHANNELS:

        columns = get_sensor_columns(
            dataframe,
            sensor,
        )

        if len(columns) != WINDOW_SIZE:
            raise ValueError(
                f"{sensor}: expected {WINDOW_SIZE} columns "
                f"but found {len(columns)}."
            )

        sensor_data = dataframe[
            columns
        ].to_numpy(dtype=np.float32)

        sensor_arrays.append(sensor_data)

    # Each individual array:
    #
    # (samples, 60)
    #
    # Stack them to:
    #
    # (samples, 6, 60)

    X = np.stack(
        sensor_arrays,
        axis=1,
    )

    if LABEL_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Label column '{LABEL_COLUMN}' not found.\n"
            f"Available columns:\n{dataframe.columns.tolist()}"
        )

    y = dataframe[
        LABEL_COLUMN
    ].to_numpy()

    return X, y