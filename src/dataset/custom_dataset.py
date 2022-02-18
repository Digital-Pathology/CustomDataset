"""
    Custom Dataset

    1) give it a directory of images, and a set of labels for those images
    2) count the regions in each image to get an initial length
    3) have getitem fetch an image according to some order
    4) include functionality for filtration
    5) include functionality for augmentation
"""

from collections import OrderedDict
import os
from typing import Any, Tuple, Union

from albumentations import Compose, BasicTransform
import numpy as np
from torch.utils.data import Dataset

from filtration import FilterManager, Filter
from unified_image_reader import Image

from . import config
from .filtration_cacheing import FiltrationCacheManager
from .label_manager import LabelManager
from .util import listdir_recursive

# TODO - figure out what happens when a region doesn't pass through a filter!!!
# TODO - add behavior for unlabaled dataset

class CustomDataset(Dataset):

    """
        Custom Dataset used to organize heirachical data
    """

    def __init__(self,
                 data_dir: str,
                 labels: Union[LabelManager, str],
                 filtration: Union[Filter, FilterManager, None] = None,
                 augmentation: Union[BasicTransform, Compose, None] = None,
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
        
        # get references to data
        self.dir = data_dir
        self._filepaths = listdir_recursive(self.dir)

        # initialize label manager
        self.label_manager = None
        if isinstance(labels, str):
            self.label_manager = LabelManager(labels)
        elif isinstance(labels, LabelManager):
            self.label_manager = labels
        else:
            raise TypeError(type(labels))

        # initialize the image_files_region_counts
        self.region_dims = kwargs.get("region_dims") or config.region_dims
        self._image_files_region_counts = OrderedDict(
            {filename: -1 for filename in os.listdir(self.dir)})
        # length is calculated lazily, and so is the cached filtration discount
        self._length = None
        self._filtration_discount = None

        # get filtration
        self.filtration = filtration
        if self.filtration:
            self.filtration_cache_manager = kwargs.get("filtration_cache_manager")
            if not self.filtration_cache_manager or \
                    not isinstance(self.filtration_cache_manager, FiltrationCacheManager):
                # TODO - default instantiation
                raise NotImplementedError("applying filtration requires a FiltrationCacheManager")

        # get augmentation
        self.augmentation = augmentation

    def __len__(self):
        """
            Set length variable to total number of regions from all images

            Returns:
                int: length (number of regions)
        """
        # TODO - update length on filtration failure???
        if self._length is None:
            self._length = self.get_number_of_regions_in_all_images(region_dims=(512, 512))
        return self._length

    def __getitem__(self, index: int):
        """
            Get region from image in database according to index.
            
            Parameters:
                index (int): index of region to pull

            Returns:
                np.ndarray: a numpy array representing the region at hashed index
        """
        # Policy for Filtration Failure:
        #   If the region doesn't pass through filtration, randomly select another region in the dataset
        region, region_passes_filtration = None, None
        while not region_passes_filtration:
            region, region_passes_filtration = self.get_filtration_status(index)
            index += 1
        return region # TODO - filtration and augmentation, return label too

    def get_filtration_status(self, index) -> bool:
        """ coordinates filtration and filtration_cache to efficiently filter regions (if necessary) """
        filename, region_identifier = self.get_region_location_from_index(index)
        if self.filtration is None:
            return True
        else:
            region = None
            region_passes_filtration = None
            # first check the filtration cache manager
            with self.filtration_cache_manager[filename] as cache:
                if region_identifier in cache:
                    region_passes_filtration = cache[region_identifier]
                # if it's not in cache then just perform filtration
                else:
                    region = self.get_region_from_image(filename, region_identifier=region_identifier)
                    region_passes_filtration = self.filtration.filter()
            return region, region_passes_filtration

    def get_region_location_from_index(self, index: int) -> Tuple[str, int]:
        """
            returns the location of the region at index

            Example:
                {a: 10, b: 25, c: 15}
                index = 15 --> (b,5)
        """
        for filename, number_of_regions in self._image_files_region_counts.items():
            if index >= number_of_regions: # region at index is in another image
                index -= number_of_regions
            else:
                return (filename, index)
        raise IndexError(f"Index of of bounds: {index=}, {len(self)=}")

    def get_region_from_image(self, filename: str, region_identifier: Union[int, Tuple[int]]):
        """
            Get region from image in database according to filename and identifier.

            Parameters:
                filename (str): filename of desired image
                region_identifier (int | Tuple[int]): either the region number or specific coordinate

            Returns:
                np.ndarray: a numpy array representing the region at specified identifier
        """
        if filename not in self._image_files_region_counts:
            raise Exception(f"file not found in dataset: {filename=}")
        return Image(filename).get_region(region_identifier, region_dims=(512, 512))

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

    def get_number_of_regions_in_all_images(self, region_dims: Tuple[int]):
        """
        Get total number of regions from all images

        Parameters:
            region_dims (Tuple[int]): Dimensions of Regions

        Returns:
            int: Number of regions in all images
        """
        for filename in self._image_files_region_counts.keys():
            filepath = os.path.join(self.dir, filename)
            filename_image = Image(filepath)
            self._image_files_region_counts[filename] = filename_image.number_of_regions(
                region_dims
            )
        # return total number of regions
        return sum(self._image_files_region_counts.values())
