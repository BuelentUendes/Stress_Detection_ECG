# Collections of helper functions that are reused across several scripts

import os


def create_directory(path: str) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
        path: Path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)

