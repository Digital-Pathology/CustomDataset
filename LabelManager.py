
import json
import csv
import os
from typing import Any, Callable
import abc

"""
label formats to support:
1) simple json
2) two-line txt files or csv???
3) directory structure
"""

class LabelManager(dict):

    def __init__(self, path: str, label_processor=None, **kwargs) -> None:
        """
        Initialize a LabelManager, a dictionary wrapper for managing labels 

        path (str): a path to the location of the labels - could be:
                    - json file {filename: label}
                    - csv (no header!)
                    - directory (label is image's relative path to parent_dir)
        """
        
        self.label_extractor: LabelExtractor = None
        if path.endswith(".json"):
            self.label_extractor = LabelExtractorJSON()
        elif path.endswith(".csv"):
            self.label_extractor = LabelExtractorCSV()
        elif path.endswith(os.path.sep):
            self.label_extractor = LabelExtractorParentDir()
        else:
            raise NotImplementedError(f'We do not support the current file extension: {path=}')

        # load labels into self
        for k, v in self.label_extractor.extract_labels(path, **kwargs).items():
            self[k] = v

        self.label_processor = label_processor
        if self.label_processor: assert (isinstance(label_processor, Callable)), f"label_processor must be callable! {type(label_processor)=}"

    def __getitem__(self, __k: str) -> Any:
        label = self.labels[__k]
        if self.label_processor:
            return self.label_processor(label)
        else:
            return label

class LabelExtractor(abc.ABC): # strategy pattern
    @abc.abstractmethod
    def extract_labels(cls, path: str, **kwargs):
        pass

class LabelExtractorJSON(LabelExtractor):
    def extract_labels(cls, path: str, **kwargs):
        with open(path, 'r') as f:
            return json.load(f)

class LabelExtractorCSV(LabelExtractor):
    def extract_labels(cls, path: str, **kwargs):
        with open(path, 'r') as f:
            reader = csv.reader(f)
            return {row[0]: row[1] for row in reader}

class LabelExtractorParentDir(LabelExtractor):
    def extract_labels(cls, path: str, **kwargs):
        return NotImplementedError()

if __name__ == "__main__":
    filename = os.path.dirname(__file__) + os.path.sep + "test_labels.json"
    label_manager = LabelManager(filename)
    print(label_manager)
