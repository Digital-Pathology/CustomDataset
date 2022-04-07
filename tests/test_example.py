
import pytest

from src.dataset import Dataset


@pytest.mark.parametrize("arg1, arg2", [
    ("test1", "test2"),
    ("test3", "test4")
])
def test_example(arg1, arg2):
    d = Dataset()
    assert (arg1 != arg2 and arg1[:-1] == arg2[:-1])
