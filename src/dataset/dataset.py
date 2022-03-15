
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
                 filtration: Union[Filter, FilterManager, None] = None,
                 augmentation: Union[albumentations.BasicTransform,
                                     albumentations.Compose, None] = None,
                 **kwargs):
        """
            Initialize Custom Dataset Object.

            Parameters:
                data_dir (str): Filepath to directory containing exclusively images (recursively)
                labels (str): Filepath to labels associated with images --> see LabelManager
                filtration: Object that applies various filters to image (optional)
                augmentation: Object that applies various augmentation techniques to image (optional)

            Returns:
                int: Number of regions in all images
        """

        # initialize dataset components
        self.dir = data_dir
        self._initialize_filepaths()
        self._initialize_label_manager(labels)
        self._initialize_filtration(filtration, **kwargs)
        self._initialize_augmentation(augmentation)

        # initialize dataset length
        self.region_dims = kwargs.get("region_dims") or config.REGION_DIMS
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

    def _initialize_filtration(self, filtration: Union[FilterManager, Filter, None], **kwargs):
        self.filtration: Union[FilterManager, Filter] = filtration
        if self.filtration:
            self.filtration_cache = kwargs.get("filtration_cache") or \
                config.DEFAULT_FILTRATION_CACHE_FILEPATH
            if isinstance(self.filtration_cache, str):
                self.filtration_cache = FiltrationCache(self.filtration_cache)
            elif not isinstance(self.filtration_cache, FiltrationCache):
                raise TypeError(type(self.filtration_cache))

    def _initialize_augmentation(self, augmentation: Union[albumentations.BasicTransform, albumentations.Compose, None]):
        self.augmentation = augmentation

    def _initialize_region_counts(self):
        self._region_counts = OrderedDict(
            {f: Image(f).number_of_regions(self.region_dims) for f in self._filepaths})

    def _initialize_region_discounts(self):
        if self.filtration is not None:
            self._region_discounts = OrderedDict()
            for image in self._region_counts.keys():
                with self.filtration_cache_manager[image] as cache:
                    self._region_discounts[image] = cache.get_regions_not_passing_filtration(
                    )

    def _initialize_length(self):
        self._length = sum(self._region_counts.values())
        if self.filtration is not None:
            self._length += sum(self._region_discounts.values())

    def _update_filepaths(self):
        self.initialize_filepaths()

    def _update_region_counts(self):
        self._initialize_region_counts()

    def _update_region_discounts(self, image: Union[str, None] = None):
        if self.filtration is None:
            return
        if image is None:
            self._initialize_region_discounts()
        else:  # update only one image's region discount
            with self.filtration_cache_manager[image] as cache:
                self._region_discounts[image] = cache.get_regions_not_passing_filtration()

    def _update_length(self, image: Union[str, None] = None):
        if image is None:
            self._initialize_length()
        else:
            if self.filtration is None:
                self._initialize_length()
            else:
                self._length -= self._region_discounts[image]
                self._update_region_discounts(image)
                self._length += self._region_discounts[image]

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
        region, label = self.get_region(index)
        region = self.augment_region(region)
        return region, label

    def get_region(self, index: int):
        """ gets region according to index using dark region indexing """
        image, region_num = self.get_region_location_from_index(index)
        region, region_passes_filtration = self.get_filtration_status(image, region_num)
        #print(f"{image=}, {region_num=}, {type(region)=}, {region_passes_filtration=}")
        if not region_passes_filtration:
            # figure out how many regions before this don't pass filtration
            prior_fails = 0
            for prior_region in range(region_num):
                region, region_passes_filtration = self.get_filtration_status(image, prior_region)
                #print(f"prior region {prior_region} --> {region_passes_filtration}")
                if not region_passes_filtration:
                    prior_fails += 1
            # count back from end for dark regions
            dark_region = self._region_counts[image] - 1
            while prior_fails >= 0:
                region, region_passes_filtration = self.get_filtration_status(image, dark_region)
                #print(f"dark region {dark_region} --> {region_passes_filtration}")
                if region_passes_filtration:
                    prior_fails -= 1
                dark_region -= 1
        return region, self.label_manager[image]
        
    def get_filtration_status(self, image, region_num) -> bool:
        """
            coordinates filtration and filtration_cache to
            efficiently filter regions (if necessary)
        """
        #print(f"get_filtration_status({image}, {region_num})")
        if self.filtration is None:
            region = Image(image).get_region(region_num, region_dims=self.region_dims)
            return region, True
        else:
            region = None
            region_passes_filtration = None
            # first check the filtration cache manager
            with self.filtration_cache_manager[image] as cache:
                #print(cache.cache)
                region_passes_filtration = cache[region_num]
                if region_passes_filtration is None: # if it's not in cache then just perform filtration
                    #print(f"{region_num=} NOT in cache")
                    region = Image(image).get_region(region_num, region_dims=self.region_dims)
                    region_passes_filtration = self.filtration.filter(region)
                    cache[region_num] = region_passes_filtration
                    self._update_length(image)
                else:
                    region = Image(image).get_region(region_num, region_dims=self.region_dims)
            return region, region_passes_filtration

    def get_region_location_from_index(self, index: int) -> Tuple[str, int]:
        """
            returns the location of the region at index

            Example:
                {a: 10, b: 25, c: 15}
                index = 15 --> (b,5)
        """
        for image in self._region_counts.keys():
            number_of_regions = self._region_counts[image]
            if self.filtration:
                number_of_regions -= self._region_discounts[image]
            if index >= number_of_regions:  # region at index is in another image
                index -= number_of_regions
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
        label_distribution = {}
        for image, region_count in self._region_counts.items():
            label = self.label_manager[image]
            if label not in label_distribution:
                label_distribution[label] = 0
            label_distribution[label] += region_count
            if self.filtration:
                label_distribution[label] -= self._region_discounts[image]
        return label_distribution
            
