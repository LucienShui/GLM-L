import torch


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return -1


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return -1


def get_data_parallel_group():
    return -1
