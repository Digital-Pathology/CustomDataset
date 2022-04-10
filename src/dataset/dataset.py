
"""
    Custom Dataset

    1) give it a directory of images, and a set of labels for those images
    2) count the regions in each image to get an initial length
    4) include functionality for filtration
    5) include functionality for augmentation
"""

from collections import OrderedDict
from typing import Any, Callable, Generator, Tuple, Union
from importlib_metadata import files

import numpy as np
from torch.utils.data import Dataset as PyTorchDataset

from filtration import FilterManager, Filter
from unified_image_reader import Image

from . import config
from .filtration_cache import FiltrationCache
from .label_manager import LabelManager
from .util import listdir_recursive

# TODO - add behavior for unlabeled dataset


class Dataset(PyTorchDataset):

    """
    Dataset is used to organize heirachical data
    """

    def __init__(self,
                 data_dir: str,
                 labels: Union[LabelManager, str],
                 augmentation: Union[Callable, None] = None,
                 filtration: Union[Filter, FilterManager, None] = None,
                 filtration_cache: Union[str, FiltrationCache, None] =
                 config.DEFAULT_FILTRATION_CACHE_FILEPATH,
                 region_dims: tuple = config.REGION_DIMS):
        """
        __init__ initializes Dataset

        :param data_dir: Filepath to directory containing exclusively images (recursively)
        :type data_dir: str
        :param labels: Filepath to labels associated with images --> see LabelManager
        :type labels: Union[LabelManager, str]
        :param augmentation: agumentation(region) returns augmented region, defaults to None
        :type augmentation: Union[Callable, None], optional
        :param filtration: Object that applies various filters to image (optional), defaults to None
        :type filtration: Union[Filter, FilterManager, None], optional
        :param filtration_cache: Reference to cache for filtration results, defaults to config.DEFAULT_FILTRATION_CACHE_FILEPATH
        :type filtration_cache: Union[str, FiltrationCache, None], optional
        :param region_dims: width and height of the images' regions, defaults to config.REGION_DIMS
        :type region_dims: tuple, optional
        """

        # initialize dataset components
        self.dir = data_dir
        self._initialize_filepaths()
        self._initialize_label_manager(labels)
        self._initialize_augmentation(augmentation)
        self._initialize_filtration(filtration, filtration_cache, region_dims)

        # initialize dataset length
        self.region_dims = region_dims
        self._initialize_region_counts()
        self._initialize_region_discounts()
        self._initialize_length()

    def _initialize_filepaths(self):
        """
        _initialize_filepaths initializes filepaths using self.dir
        """
        self._filepaths = listdir_recursive(self.dir)

    def _initialize_label_manager(self, labels: Union[LabelManager, str, None]):
        """
        _initialize_label_manager initializes label management

        :param labels: labels to be extracted
        :type labels: Union[LabelManager, str, None]
        :raises TypeError: if labels is an unsupported type
        """
        self.label_manager = None
        if isinstance(labels, str):
            self.label_manager = LabelManager(labels)
        elif isinstance(labels, LabelManager):
            self.label_manager = labels
        else:
            raise TypeError(type(labels))

    def _initialize_filtration(self,
                               filtration: Union[FilterManager, Filter, None],
                               filtration_cache: Union[FiltrationCache, str, None],
                               region_dims: tuple):
        """
        _initialize_filtration initializes filtration if applicable

        :param filtration: filtration to be applied to regions in the dataset's images
        :type filtration: Union[FilterManager, Filter, None]
        :param filtration_cache: where/how filtration work should be recorded
        :type filtration_cache: Union[FiltrationCache, str, None]
        :param region_dims: dimensions of the images' regions
        :type region_dims: tuple
        :raises Exception: if region dimensions are conflicting given both region dimensions and existing filtration cache
        :raises TypeError: if the filtration cache is of unsupported type
        """
        self.filtration: Union[FilterManager, Filter] = filtration
        if self.filtration:
            self.filtration_cache = filtration_cache
            if isinstance(self.filtration_cache, str):
                self.filtration_cache = \
                    FiltrationCache(self.filtration_cache,
                                    region_dims=region_dims)
            elif isinstance(self.filtration_cache, FiltrationCache):
                if region_dims != self.filtration_cache.region_dims:
                    raise Exception(
                        f"region_dims must be the same: \
                            {region_dims=}, {self.filtration_cache.region_dims=}")
            else:
                raise TypeError(type(self.filtration_cache))
            self._preprocess_images()

    def _initialize_augmentation(self, augmentation: Union[Callable, None]):
        """
        _initialize_augmentation initializes augmentation

        :param augmentation: function for maintainability
        :type augmentation: Union[Callable, None]
        """
        self.augmentation = augmentation

    def _initialize_region_counts(self):
        """
        _initialize_region_counts initializes region counts based on self._filepaths
        """
        self._region_counts = OrderedDict(
            {f: Image(f).number_of_regions(self.region_dims) for f in self._filepaths})

    def _initialize_region_discounts(self):
        """
        _initialize_region_discounts initializes region discounts based on filtration/filtration_cache, if applicable
        """
        self._region_discounts = OrderedDict({
            img: None for img in self._region_counts.keys()
        })
        if self.filtration is not None:
            for img in self._region_discounts.keys():
                metadata = self.filtration_cache.get_metadata(
                    self.filtration,
                    img
                )
                self._region_discounts[img] = metadata["_image_dark_regions_count"]

    def _initialize_length(self):
        """
        _initialize_length calculates the length of the dataset based on region counts/discounts
        """
        self._length = sum(self._region_counts.values())
        if self.filtration is not None:
            self._length -= sum(self._region_discounts.values())

    def _preprocess_images(self):
        """
        _preprocess_images preprocesses images w.r.t. filtration and the filtration cache
        """
        for i, image in enumerate(self._filepaths):
            #print(f"Preprocessing {i}/{len(self._filepaths)} {image}")
            self.filtration_cache.preprocess(
                self.filtration, image, overwrite=False)

    def __len__(self) -> int:
        """
        __len__ override for Pytorch dataset

        :return: the number of regions in all images
        :rtype: int
        """
        return self._length

    def __getitem__(self, index: int) -> np.ndarray:
        """
        __getitem__ returns a region from the dataset

        :param index: the region identifier
        :type index: int
        :return: the region in question
        :rtype: numpy.ndarray
        """
        image, region_num = self.get_region_location_from_index(index)
        label = self.label_manager[image]
        region = self.get_region(image, region_num)
        region = self.augment_region(region)
        return region, label

    def get_region_location_from_index(self, index: int) -> Tuple[str, int]:
        """
        get_region_location_from_index returns the image, region_num location of the region at dataset[i]

        :param index: dataset[i]
        :type index: int
        :raises IndexError: if index is out of bounds
        :return: the location of the region at dataset[i]
        :rtype: Tuple[str, int]
        """
        for image, region_count in self._region_counts.items():
            if self.filtration:
                region_count -= self._region_discounts[image]
            if index >= region_count:  # region at index is in another image
                index -= region_count
            else:
                return (image, index)
        raise IndexError(f"Index of of bounds: {index=}, {len(self)=}")

    def augment_region(self, region: np.ndarray) -> np.ndarray:
        """
        augment_region augments a region using self.augmentation

        :param region: the region to be (potentially) augmented
        :type region: np.ndarray
        :return: the (potentially) augmented region
        :rtype: np.ndarray
        """
        if self.augmentation:
            return self.augmentation(region)
        return region

    def get_label(self, filename: str) -> Any:
        """
        get_label returns the label associated with a certain filename

        :param filename: the filename in question
        :type filename: str
        :return: the label for the filename in question
        :rtype: Any
        """
        # TODO - add expansion for region-specific labels?
        return self.label_manager[filename]

    def get_region(self, filename: str, region_num: int) -> np.ndarray:
        """
        get_region returns the region at location region num in filename, applying filtration and filtration cache if necessary

        :param filename: the filename from which to get the region
        :type filename: str
        :param region_num: the number of the region in the image at filename
        :type region_num: int
        :return: the region in question
        :rtype: np.ndarray
        """
        if self.filtration is None:
            pass  # region num is exactly what it seems like
        else:  # check filtration cache for proper target region
            #print(region_num, type(region_num))
            region_num = int(self.filtration_cache.get_status(
                self.filtration,
                filename,
                region_num
            )[1])
            #print(region_num, type(region_num))
        return Image(filename).get_region(region_num)

    def get_label_distribution(self) -> dict:
        """
        get_label_distribution gives the count of regions belonging to each label - assumes region label is region's image's label

        :return: a dictionary representing the counts of the images' labels
        :rtype: dict
        """
        label_distribution = {}
        for image, region_count in self._region_counts.items():
            label = self.label_manager[image]
            if label not in label_distribution:
                label_distribution[label] = 0
            label_distribution[label] += region_count
            if self.filtration:
                label_distribution[label] -= self._region_discounts[image]
        return label_distribution

    def iterate_by_file(self) -> Generator[Tuple[str, Any, Generator], None, None]:
        """
        iterate_by_file allows for users to iterate over region in an image given the filename and the label

        :yield: the filename, the label, and an iterator for the regions in the image
        :rtype: Tuple[str, Any, Generator]
        """
        """ 
        dataset = Dataset()
        for (filename, label, regions) in dataset.iterate_by_file():
            regions = list(regions)
            continue
        """
        def regions_generator(filename: str) -> Generator[np.ndarray, None, None]:
            """
            regions_generator iterates over the regions in an image

            :param filename: the image to iterate over
            :type filename: str
            :yield: regions
            :rtype: numpy.ndarray
            """
            for i in range(self._region_counts[filename] - self._region_discounts[filename]):
                yield self.get_region(filename, i)
        for filename in self._filepaths:
            yield filename, self.get_label(filename), regions_generator(filename)
