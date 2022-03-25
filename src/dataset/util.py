import os
from typing import List


def listdir_recursive(path: str) -> List[str]:
    """ list files (not directories) recursively from path """
    files = []
    walk = os.walk(path)
    for (directory_pointer, _, file_nodes) in walk:
        files += [
            os.path.join(directory_pointer, file_node)
            for file_node in file_nodes
        ]
    return files


def path_without_basename(path: str) -> str:
    return os.path.dirname(path)
