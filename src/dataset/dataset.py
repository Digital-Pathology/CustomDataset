
"""
    Custom Dataset

    1) give it a directory of images, and a set of labels for those images
    2) count the regions in each image to get an initial length
    4) include functionality for filtration
    5) include functionality for augmentation
"""

from collections import OrderedDict
from typing import Any, Tuple, Union

import albumentations
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
        Custom Dataset used to organize heirachical data
    """

    def __init__(self,
                 data_dir: str,
                 labels: Union[LabelManager, str],
                 augmentation: Union[albumentations.BasicTransform,
                                     albumentations.Compose, None] = None,
                 filtration: Union[Filter, FilterManager, None] = None,
                 filtration_cache: Union[str, FiltrationCache, None] = \
                     config.DEFAULT_FILTRATION_CACHE_FILEPATH,
                 region_dims: tuple = config.REGION_DIMS):
        """
            Initialize Custom Dataset Object.

            Parameters:
                data_dir (str): Filepath to directory containing exclusively images (recursively)
                labels (str): Filepath to labels associated with images --> see LabelManager
                filtration: Object that applies various filters to image (optional)
                filtration_cache: Reference to cache for filtration results
                augmentation: Applies various augmentation techniques to image (optional)
                region_dims (tuple[int, int]): width and height of the images' regions

            Returns:
                int: Number of regions in all images
        """

        # initialize dataset components
        self.dir = data_dir
        self._initialize_filepaths()
        self._initialize_label_manager(labels)
        self._initialize_augmentation(augmentation)
        self._preprocess_images()
        self._initialize_filtration(filtration, filtration_cache, region_dims)

        # initialize dataset length
        self.region_dims = region_dims
        self._initialize_region_counts()
        self._initialize_region_discounts()
        self._initialize_length()

    def _initialize_filepaths(self):
        self._filepaths = listdir_recursive(self.dir)

    def _initialize_label_manager(self, labels: Union[LabelManager, str, None]):
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
        self.filtration: Union[FilterManager, Filter] = filtration
        if self.filtration:
            self.filtration_cache = filtration_cache
            if isinstance(self.filtration_cache, str):
                self.filtration_cache = \
                    FiltrationCache(self.filtration_cache, region_dims=region_dims)
            elif isinstance(self.filtration_cache, FiltrationCache):
                if region_dims != self.filtration_cache.region_dims:
                    raise Exception(
                        f"region_dims must be the same: \
                            {region_dims=}, {self.filtration_cache.region_dims=}")
            else:
                raise TypeError(type(self.filtration_cache))

    def _initialize_augmentation(self,
                                 augmentation: \
                                    Union[albumentations.BasicTransform, albumentations.Compose, None]):
        self.augmentation = augmentation

    def _initialize_region_counts(self):
        self._region_counts = OrderedDict(
            {f: Image(f).number_of_regions(self.region_dims) for f in self._filepaths})

    def _initialize_region_discounts(self):
        self._region_discounts = OrderedDict({
            img: None for img in self._region_counts.keys()
        })
        if self.filtration is not None:
            for img in self._region_discounts.keys():
                self._region_discounts[img] = self.filtration_cache.get_metadata(
                    self.filtration,
                    img
                )["_image_dark_regions_count"]

    def _initialize_length(self):
        self._length = sum(self._region_counts.values())
        if self.filtration is not None:
            self._length -= sum(self._region_discounts.values())

    def _preprocess_images(self):
        for i, image in enumerate(self._filepaths):
            print(f"Preprocessing {i}/{len(self._filepaths)} {image}")
            self.filtration_cache.preprocess(self.filtration, image, overwrite=False)

    def __len__(self):
        return self._length

    def __getitem__(self, index: int):
        """
            Get region from image in database according to index and augment it, if applicable.

            Parameters:
                index (int): index of region to pull

            Returns:
                np.ndarray: a numpy array representing the region at hashed index

            Index Mapping
                Example: {image:region_count} = {'a':10, 'b':25, 'c':15}
                Find: dataset[40]
                With no Filtration: c[5]
                With Filtration: See README for details
        """
        image, region_num = self.get_region_location_from_index(index)
        label = self.label_manager[image]
        region = Image(image).get_region(region_num)
        region = self.augment_region(region)
        return region, label

    def get_region_location_from_index(self, index: int) -> Tuple[str, int]:
        """
            returns the location of the region at index

            Example:
                {a: 10, b: 25, c: 15}
                index = 15 --> (b,5)
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
        """ augments a region using self.augmentation """
        if self.augmentation:
            return self.augmentation(image=region)["image"]
        else:
            return region

    def get_label(self, filename: str) -> Any:
        """ returns the label associated with a certain filename """
        # TODO - add expansion for region-specific labels?
        return self.label_manager[filename]

    def get_label_distribution(self):
        """ gives the count of regions belonging to each label -
            assumes region label is region's image's label """
        label_distribution = {}
        for image, region_count in self._region_counts.items():
            label = self.label_manager[image]
            if label not in label_distribution:
                label_distribution[label] = 0
            label_distribution[label] += region_count
            if self.filtration:
                label_distribution[label] -= self._region_discounts[image]
        return label_distribution
