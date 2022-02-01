
from collections import OrderedDict
import os
from types import NoneType
from typing import Tuple, Union

import numpy as np
from torch.utils.data import Dataset, DataLoader

import AugmentationManager
import Augmentation
from Filtration import FilterManager, Filter
import LabelManager
import UnifiedImageReader

"""
    Custom Dataset

    1) give it a directory of images, and a set of labels for those images
    2) count the regions in each image to get an initial length
    3) have getitem fetch an image according to some order
    4) include functionality for filtration
    5) include functionality for augmentation
"""

# TODO - add augmentations
# TODO - figure out what happens when a region doesn't pass through a filter!!!

class CustomDataset(Dataset):

    def __init__(self, 
            data_dir: str, 
            labels: Union[LabelManager.LabelManager, str],
            filtration: Union[Filter.Filter, FilterManager.FilterManager, NoneType], 
            augmentation: Union[Augmentation.Augmentation, AugmentationManager.AugmentationManager, NoneType]) -> None:
        """
        Initialize Custom Dataset Object.

        Parameters:
            data_dir (str): Filepath to directory of images (and only images)
            labels (str): Filepath to labels associated with images
            filtration (FilterManager): FilterManager that applies various filters to the image (optional)
            augmentation (AugmentationManager): FilterManager that applies various augmentation techniques to the image (optional)

        Returns:
            int: Number of regions in all images
        """
        self.dir = data_dir
        self.label_manager = LabelManager(labels)

        self._image_files_region_counts = OrderedDict({filename:-1 for filename in os.listdir(self.dir)})
        self._length = None
        
        self.filtration = filtration

        self.augmentation = augmentation
    
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
            filename_image = UnifiedImageReader.Image(filepath)
            self._image_files_region_counts[filename] = filename_image.number_of_regions(region_dims)
        # return total number of regions
        return sum(self._image_files_region_counts.values())
    
    def __len__(self):
        """
        Set length variable to total number of regions from all images

        Returns:
            int: length (number of regions)
        """
        if self._length is None:
            self._length = self.get_number_of_regions_in_all_images(region_dims=(512, 512))
        return self._length

    def __getitem__(self, index: int):
        # TODO - return the label as well - how is that supposed to work?
        """
        Get region from image in database according to index.
        Example: getitem(15) on {a: 10, b: 25, c: 15} skips over all 10 regions of a, and gets region b[5]

        Parameters:
            index (int): index of region to pull

        Returns:
            np.ndarray: a numpy array representing the region at hashed index
        """
        for filename, number_of_regions in self._image_files_region_counts.items():
            if index < number_of_regions: # region at index is in image at filename
                filename_image = UnifiedImageReader(filename)
                region = filename_image.get_region(region_identifier=index, region_dims=(512, 512))
                return self.process_region(region)
            else: # region at index is in another file
                index -= number_of_regions
        raise IndexError(f"Index of of bounds: {index=}, {len(self)=}")

    def get_region_from_image(self, filename: str, region_identifier: Union[int, tuple]):
        if filename not in self._image_files_region_counts:
            raise Exception(f"file not found: {filename=}")
        return UnifiedImageReader.Image(filename).get_region(region_identifier, region_dims=(512,512))

    def process_region(self, region: np.ndarray):
        """
        Perform optional filtration and/or augmentation to a region.

        Parameters:
            region (np.ndarray): numpy array representing the region

        Returns:
            np.ndarray: a numpy array representing the region after performing optional processing
        """
        if self.filtration is not None:
            # apply filter to image???
            raise NotImplementedError()
        if self.augmentation is not None:
            # apply augmentation???
            raise NotImplementedError()
        return region

#class UnlabeledCustomDataset(CustomDataset):
#
#    def __init__(self, data_dir) -> None:
#        super().__init__(data_dir, None)
#        raise NotImplementedError()
#    
#    def __getitem__(self, index): # dont return a label, and return image as numpy array???
#        return super().__getitem__(index)