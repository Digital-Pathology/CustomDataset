
"""
    Utility functions, etc. for the custom dataset
"""

# stdlib
import contextlib
from itertools import repeat
import os
import threading
from typing import Any, Dict, List, NewType, Optional, Tuple, Union

# in-house imports
from filtration import Filter, FilterManager

FilePath = NewType('Filepath', str)
FiltrationRepr = NewType('FiltrationRepr', Union[Filter, FilterManager, str])
FiltrationStatus = NewType('FiltrationStatus', Tuple[int, int, bool])  # index, index_target, filtration_status
FiltrationCacheMetadata = NewType(
    'FiltrationCacheMetadata', Optional[Dict[str, Any]])


class UnsupportedFileType(Exception):
    """
    UnsupportedFileType is raised when a file extension can't be parsed natively
    """


def listdir_recursive(path: FilePath) -> List[FilePath]:
    """
    listdir_recursive lists files (not directories) recursively from path

    :param path: the path to the directory whose files should be listed recursively
    :type path: FilePath
    :return: a list of filepaths relative to path
    :rtype: List[FilePath]
    """
    files = []
    walk = os.walk(path)
    for (directory_pointer, _, file_nodes) in walk:
        files += [
            os.path.join(directory_pointer, file_node)
            for file_node in file_nodes
        ]
    return files


class ThreadingLock(contextlib.AbstractContextManager):
    """ 
        A wrapper on threading.Lock that implements a context manager so that when the context closes the lock will unlock

        Example:
            with status_lock as permission: # will hang until it gets permission
                # do things
    """

    def __init__(self):
        self.lock = threading.Lock()

    def __enter__(self, *args, **kwargs):
        while not self.lock.acquire():
            pass  # wait for my turn
        return True

    def __exit__(self, *args, **kwargs):
        self.lock.release()


def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    """ https://stackoverflow.com/a/53173433/13747259 """
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)


def apply_args_and_kwargs(fn, args, kwargs):
    """ https://stackoverflow.com/a/53173433/13747259 """
    return fn(*args, **kwargs)
