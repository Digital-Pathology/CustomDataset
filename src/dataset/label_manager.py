"""
label formats to support:
1) simple json
2) two-line txt files or csv???
3) directory structure
"""

import abc
import csv
import json
import os
from typing import Any, Callable

from . import config
from .util import listdir_recursive, path_without_basename


class LabelManager:

    """
    A dictionary wrapper for managing labels
    """

    def __init__(self, path: str, **kwargs) -> None:
        
        """
            Initialize a LabelManager

                Parameters

                    path (str): a path to the location of the labels - could be:
                                - json file {filename: label}
                                - csv (no header!)
                                - directory (label is image's relative path to parent_dir)
            
                Optional kwargs

                    label_extractor (LabelExtractor): overrides automatic selection
                    label_preprocessor (Callable): allows user to preprocess key for label lookup
                    label_postprocessor (Callable): allows user to postprocess label inside LabelManager
        """

        super().__init__()

        # get label extractor or choose based on path
        self.label_extractor = kwargs.get("label_extractor")
        if self.label_extractor is None:
            if path.endswith(".json"):
                self.label_extractor = LabelExtractorJSON()
            elif path.endswith(".csv"):
                self.label_extractor = LabelExtractorCSV()
            elif os.path.isdir(path):
                self.label_extractor = LabelExtractorParentDir()
            else:
                raise NotImplementedError(
                    f'We do not support the current file extension: {path=}')
        else:
            if not isinstance(self.label_extractor, LabelExtractor):
                raise TypeError(type(self.label_extractor))

        # get labels from label_extractor
        self.labels = self.label_extractor.extract_labels(path, **kwargs)

        # label preprocessing
        #   user can have the path of the file preprocessed before indexing the extracted labels
        self.label_preprocessor = kwargs.get("label_preprocessor")
        if self.label_preprocessor is not None and not isinstance(self.label_preprocessor, Callable):
            raise TypeError(self.label_preprocessor)

        # label postprocessor
        #   user can have the extracted label postprocessed inside the label manager
        self.label_postprocessor = kwargs.get("label_postprocessor")
        if self.label_postprocessor is not None and not isinstance(self.label_postprocessor, Callable):
            raise TypeError(type(self.label_postprocessor))
        self.error_if_no_label = kwargs.get("error_if_no_label") or config.LABEL_MANAGER_IF_NO_LABEL

    def __getitem__(self, key: str) -> Any:
        # preprocess if applicable
        if self.label_preprocessor is not None:
            key = self.label_preprocessor(key)
        label = self.labels.get(key)
        # postprocess if applicable
        if self.label_postprocessor is not None:
            label = self.label_postprocessor(label)
        if label is None:
            raise IndexError(key)
        return label


class LabelExtractor(abc.ABC):  # strategy pattern
    """ Strategy Pattern --> extracts labels from path for dictionary-based lookup """
    @staticmethod
    @abc.abstractmethod
    def extract_labels(path: str, **kwargs):
        """ extracts labels from path for dictionary-based lookup """


class LabelExtractorJSON(LabelExtractor):
    """ labels in json file """
    @staticmethod
    def extract_labels(path: str, **kwargs):
        """ labels are inside of a json file at path of structure {key: label, ...} """
        with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)


class LabelExtractorCSV(LabelExtractor):
    """ labels in csv file """
    @staticmethod
    def extract_labels(path: str, **kwargs):
        """ labels are inside of a csv file at path of structure (each line) <key><sep><label> """
        with open(path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            return {row[0]: row[1] for row in reader}


class LabelExtractorParentDir(LabelExtractor):
    """ labels represented by relative path """
    @staticmethod
    def extract_labels(path: str, **kwargs):
        """ labels are path relative to path arg (label_postprocessor recommended) """
        files = listdir_recursive(path)
        return {f: path_without_basename(f) for f in files}
