
"""
    Label inference for the custom dataset
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from . import label_extractor
from . import util


class LabelManager:
    """
     A dictionary wrapper for managing labels

    :raises NotImplementedError: when a given file extension cannot be parsed natively
    :raises TypeError: when the label_extractor isn't a LabelExtractor
    :raises TypeError: when label_preprocessor isn't Callable
    :raises TypeError: when label_postprocessor isn't Callable
    :raises IndexError: when a key doesn't have a value
    """

    def __init__(self,
                 path: util.FilePath,
                 label_extraction: Optional[label_extractor.LabelExtractor] = None,
                 label_preprocessor: Optional[Callable] = None,
                 label_postprocessor: Optional[Callable] = None,
                 error_if_no_labels: bool = True) -> None:
        """
        __init__ initializes a LabelManager

        :param path: A filepath from where labels can be inferred
        :type path: util.FilePath
        :param label_extractor: how labels are parsed (if None, uses file extension), defaults to None
        :type label_extractor: Optional[LabelExtractor], optional
        :param label_preprocessor: a function for preprocessing the key used to get the label, defaults to None
        :type label_preprocessor: Optional[Callable], optional
        :param label_postprocessor: a function for postprocessing labels after they are indexed, defaults to None
        :type label_postprocessor: Optional[Callable], optional
        :param error_if_no_labels: whether LabelManager should throw IndexError, defaults to True
        :type error_if_no_labels: bool, optional
        :raises util.UnsupportedFileType: when a given file extension cannot be parsed natively
        :raises TypeError: when the label_extractor isn't a LabelExtractor
        :raises TypeError: when label_preprocessor isn't Callable
        :raises TypeError: when label_postprocessor isn't Callable
        """

        super().__init__()

        # get labels with a LabelExtractor
        if label_extraction is None:
            if path.endswith(".json"):
                label_extraction = label_extractor.LabelExtractorJSON()
            elif path.endswith(".csv"):
                label_extraction = label_extractor.LabelExtractorCSV()
            elif os.path.isdir(path):
                label_extraction = label_extractor.LabelExtractorParentDir()
            else:
                raise util.UnsupportedFileType(
                    f'We do not support that file extension: {path=}')
        else:
            if not isinstance(label_extraction, label_extractor.LabelExtractor):
                raise TypeError(type(label_extraction))
        self.labels = label_extraction.extract_labels(path)

        # label preprocessing
        #   user can have the path of the file preprocessed before indexing the extracted labels
        self.label_preprocessor = label_preprocessor
        if self.label_preprocessor is not None and not isinstance(
                self.label_preprocessor, Callable):
            raise TypeError(self.label_preprocessor)

        # label postprocessor
        #   user can have the extracted label postprocessed inside the label manager
        self.label_postprocessor = label_postprocessor
        if self.label_postprocessor is not None and not isinstance(
                self.label_postprocessor, Callable):
            raise TypeError(type(self.label_postprocessor))
        self.error_if_no_label = error_if_no_labels

    def __getitem__(self, key: str) -> Any:
        """
        __getitem__ indexes the label manager's labels

        :param key: the key of the label to index
        :type key: str
        :raises IndexError: when the key does not have a corresponding value (if self.error_if_no_label)
        :return: the label
        :rtype: Any
        """
        # preprocess if applicable
        if self.label_preprocessor is not None:
            key = self.label_preprocessor(key)
        label = self.labels.get(key)
        # postprocess if applicable
        if label is None and self.error_if_no_label:
            raise IndexError(key)
        label = self.label_postprocessor(label)
        return label
