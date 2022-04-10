
"""
    Utility functions, etc. for the custom dataset
"""

# stdlib
import os
from typing import Any, Dict, List, NewType, Optional, Tuple, Union

# in-house imports
from filtration import Filter, FilterManager

FilePath = NewType('Filepath', str)
FiltrationRepr = NewType('FiltrationRepr', Union[Filter, FilterManager, str])
FiltrationStatus = NewType('FiltrationStatus', Tuple[int, int, bool])
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
