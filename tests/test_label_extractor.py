
import pytest

from src.dataset.label_extractor import \
    LabelExtractorNoLabels, \
    LabelExtractorCSV, \
    LabelExtractorJSON, \
    LabelExtractorParentDir

CORRECT_LABELS = {
    "dog.tif": "cute",
    "test_image.tif": "not_cute",
    "test_image.tiff": "not_cute"
}


def test_label_extractor_no_labels():
    pass


def test_label_extractor_csv():
    pass


def test_label_extractor_json():
    pass


def test_label_extractor_parent_dir():
    pass

# @pytest.mark.parametrize("arg1, arg2", [
#    ("test1", "test2"),
#    ("test3", "test4")
# ])
# def test_example(arg1, arg2):
#    assert (arg1 != arg2 and arg1[:-1] == arg2[:-1])
