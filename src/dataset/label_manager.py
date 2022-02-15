"""
label formats to support:
1) simple json
2) two-line txt files or csv???
3) directory structure
"""

import json
import csv
import os
from typing import Any, Callable, Union
import abc


class LabelManager(dict):
    """
    A dictionary wrapper for managing labels
    """

    def __init__(self, path: str, label_processor: Union[Callable, None] = None, **kwargs) -> None:
        """
        Initialize a LabelManager

        path (str): a path to the location of the labels - could be:
                    - json file {filename: label}
                    - csv (no header!)
                    - directory (label is image's relative path to parent_dir)
        label_processor (Callable): Object that processes labels if provided (optional)
        """

        self.label_extractor: LabelExtractor = None
        if path.endswith(".json"):
            self.label_extractor = LabelExtractorJSON()
        elif path.endswith(".csv"):
            self.label_extractor = LabelExtractorCSV()
        elif path.endswith(os.path.sep):
            self.label_extractor = LabelExtractorParentDir()
        else:
            raise NotImplementedError(
                f'We do not support the current file extension: {path=}')

        for key, value in self.label_extractor.extract_labels(path, **kwargs).items():
            self[key] = value

        self.label_processor = label_processor
        if self.label_processor:
            assert (isinstance(label_processor, Callable)
                    ), f"label_processor must be callable! {type(label_processor)=}"

    def __getitem__(self, __k: str) -> Any:
        return self.label_processor(self[__k]) if self.label_processor else self[__k]


class LabelExtractor(abc.ABC):  # strategy pattern
    @staticmethod
    @abc.abstractmethod
    def extract_labels(path: str, **kwargs):
        pass


class LabelExtractorJSON(LabelExtractor):
    @staticmethod
    def extract_labels(path: str, **kwargs):
        with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)


class LabelExtractorCSV(LabelExtractor):
    @staticmethod
    def extract_labels(path: str, **kwargs):
        with open(path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            return {row[0]: row[1] for row in reader}


class LabelExtractorParentDir(LabelExtractor):
    @staticmethod
    def extract_labels(path: str, **kwargs):
        return NotImplementedError()


if __name__ == "__main__":
    filename = os.path.dirname(__file__) + os.path.sep + "test_labels.json"
    label_manager = LabelManager(filename)
    print(label_manager)
