
"""
    An automatic (or overloaded) label inference class
"""

import abc
from collections import defaultdict
import csv
from email.policy import default
import json
import os

from . import util


class LabelExtractor(abc.ABC):  # strategy pattern
    """ Strategy Pattern --> extracts labels from path for dictionary-based lookup """

    @staticmethod
    @abc.abstractmethod
    def extract_labels(path: str):
        """ extracts labels from path for dictionary-based lookup """


class LabelExtractorNoLabels(LabelExtractor):
    class DefaultDictWithGet(defaultdict):
        def get(self, *args, **kwargs):
            return 'LabelExtractorNoLabels'

    @staticmethod
    def extract_labels(path: str):
        """ returns None for all labels """
        return LabelExtractorNoLabels.DefaultDictWithGet()


class LabelExtractorJSON(LabelExtractor):
    """ labels in json file """

    @staticmethod
    def extract_labels(path: str):
        """ labels are inside of a json file at path of structure {key: label, ...} """
        with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)


class LabelExtractorCSV(LabelExtractor):
    """ labels in csv file """

    @staticmethod
    def extract_labels(path: str):
        """ labels are inside of a csv file at path of structure (each line) <key><sep><label> """
        with open(path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            return {row[0]: row[1] for row in reader}


class LabelExtractorParentDir(LabelExtractor):
    """ labels represented by relative path """

    @staticmethod
    def extract_labels(path: str):
        """ labels are path relative to path arg (label_postprocessor recommended) """
        files = util.listdir_recursive(path)
        return {f: os.path.dirname(f) for f in files}
