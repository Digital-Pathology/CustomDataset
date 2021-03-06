
"""
    The Dataset object aggregates information about your whole-slide images,
    their labels, any filtration that should be applied to regions thereof,
    and any augmentation functions that should be applied to the regions as
    they are fetched from the disk. This class inherits from PyTorch's
    dataset.
"""

from collections import OrderedDict
from multiprocessing import Pool
import os
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import Dataset as PyTorchDataset
from tqdm import tqdm as loadingbar

from filtration import FilterManager, Filter
from unified_image_reader import Image

from . import config
from .filtration_cache import FiltrationCache
from .label_manager import LabelManager
from .util import listdir_recursive, starmap_with_kwargs

# TODO - add behavior for unlabeled dataset


class Dataset(PyTorchDataset):

    """
    Dataset implements automatic label inference, optional augmentation, and optional dynamic filtration.
    """

    def __init__(self,
                 data: str,
                 labels: Union[LabelManager, str],
                 augmentation: Union[Callable, None] = None,
                 filtration: Union[Filter, FilterManager, None] = None,
                 filtration_cache: Union[str, FiltrationCache, None] =
                 config.DEFAULT_FILTRATION_CACHE_FILEPATH,
                 filtration_preprocess: Optional[dict] = None,
                 filtration_preprocess_loadingbars: bool = False,
                 filtration_preprocess_lazy: bool = False,
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
        :param filtration_preprocess: A dictionary passed as kwargs to filtration preprocessing functions, defaults to None
        :type filtration_preprocess: Optional[dict]
        :param filtration_preprocess_loadingbars: whether to display loadingbars during preprocessing, defaults to False
        :type filtration_preprocess: bool
        :param filtration_preprocess_lazy: whether to avoid filtration preprocessing during dataset initialization, defaults to False
        :type filtration_preprocess_lazy: bool
        :param region_dims: width and height of the images' regions, defaults to config.REGION_DIMS
        :type region_dims: tuple, optional
        """

        # initialize dataset components
        self._initialize_filepaths(data)
        self._initialize_label_manager(labels)
        self._initialize_augmentation(augmentation)
        self._initialize_filtration(filtration, filtration_cache, region_dims, filtration_preprocess_lazy,
                                    filtration_preprocess, filtration_preprocess_loadingbars)

        # initialize dataset length
        self.region_dims = region_dims
        self._initialize_region_counts()
        self._initialize_region_discounts()
        self._initialize_length()

    def _initialize_filepaths(self, data):
        """
            _initialize_filepaths
        """
        self._filepaths = None
        if isinstance(data, str):  # expecting a path
            if not os.path.exists(data):
                raise Exception(
                    f"Excepted a path but {data} does not exist as a path")
            if os.path.isdir(data):
                self._filepaths = listdir_recursive(data)
            elif os.path.isfile(data):
                self._filepaths = [data]
        elif isinstance(data, [list, tuple]):
            self._filepaths = data
            for filepath in self._filepaths:
                if not os.path.isfile(filepath):
                    raise Exception(f"{filepath} should be a path to an image")
        else:
            raise TypeError(f"Didn't expect {type(data)=}, {data=}")

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
                               region_dims: tuple,
                               filtration_preprocess_lazy: bool,
                               filtration_preprocess: Optional[dict],
                               filtration_preprocess_loadingbars: bool):
        """
        _initialize_filtration initializes filtration if applicable

        :param filtration: filtration to be applied to regions in the dataset's images
        :type filtration: Union[FilterManager, Filter, None]
        :param filtration_cache: where/how filtration work should be recorded
        :type filtration_cache: Union[FiltrationCache, str, None]
        :param region_dims: dimensions of the images' regions
        :type region_dims: tuple
        :param filtration_preprocess: kwargs for filtration preprocessing for measure-based top-n filtration
        :type filtration_preprocess: Optional[dict]
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
            if not filtration_preprocess_lazy:
                self._preprocess_images(
                    filtration_preprocess, filtration_preprocess_loadingbars)

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
            img: 0 for img in self._region_counts.keys()
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

    def _preprocess_images(self, filtration_preprocess: Optional[dict], loadingbars: bool) -> None:
        """
        _preprocess_images preprocesses images w.r.t. filtration and the filtration cache

        :param filtration_preprocess: additional arguments to pass to self.filtration_cache.preprocess such as for measure-based top-n filtration
        :type filtration_preprocess: Optional[dict]
        """
        if not config.DATASET_FILTRATION_PREPROCESSING_MULTIPROCESSING:
            iterator = enumerate(self._filepaths)
            if loadingbars:
                iterator = loadingbar(iterator, total=len(self._filepaths))
            for i, image in iterator:
                self.filtration_cache.preprocess(
                    self.filtration, image, overwrite=False, loadingbars=loadingbars, **(filtration_preprocess or {}))
        else:
            try:
                pool = Pool()
                starmap_with_kwargs(pool, self.filtration_cache.preprocess,
                                    args_iter=((self.filtration, image)
                                               for image in self._filepaths),
                                    kwargs_iter=({
                                        "overwrite": False,
                                        "loadingbars": False,
                                        **(filtration_preprocess or {})
                                    } for _ in self._filepaths)
                                    )
            except Exception as e:
                pool.close()
                pool.join()
                raise e

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
        for image in self._filepaths:
            region_count = self.number_of_regions(image)
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
            region_num = int(self.filtration_cache.get_status(
                self.filtration,
                filename,
                region_num
            )[1])
        region = Image(filename).get_region(region_num)
        region = self.augment_region(region)
        return region

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

    def iterate_by_file(self, as_pytorch_datasets=False) -> Generator[Tuple[str, Any, Generator], None, None]:
        """
        iterate_by_file allows for users to iterate over regions in an image given the filename and the label

        :param as_pytorch_datasets: Whether to return pytorch datasets instead of the normal generators, defaults to False
        :type as_pytorch_datasets: bool
        :yield: the filename, the label, and an iterator for the regions in the image
        :rtype: Tuple[str, Any, Generator]
        """

        if not as_pytorch_datasets:
            def regions_generator(filename: str) -> Generator[np.ndarray, None, None]:
                """
                regions_generator iterates over the regions in an image

                :param filename: the image to iterate over
                :type filename: str
                :yield: regions
                :rtype: numpy.ndarray
                """
                for i in range(self.number_of_regions(filename)):
                    yield self.get_region(filename, i)
            for filename in self._filepaths:
                yield filename, self.get_label(filename), regions_generator(filename)
        else:
            class SingleFileDataset(PyTorchDataset):
                def __init__(self, base_dataset, filename) -> None:
                    self.base_dataset = base_dataset
                    self.filename = filename

                def __getitem__(self, index):
                    return self.base_dataset.get_region(self.filename, index)

                def __len__(self):
                    return self.base_dataset.number_of_regions(self.filename)
            for filename in self._filepaths:
                yield filename, self.get_label(filename), SingleFileDataset(base_dataset=self, filename=filename)

    def number_of_regions(self, filename: Optional[str] = None) -> int:
        """
        number_of_regions get the number of regions in the dataset or in a single image managed by the dataset

        :param filename: An optional parameter which will get the number of regions in that image (if it's in the dataset) instead of the number of regions in the entire dataset, defaults to None
        :type filename: Optional[str]
        :return: the number of regions in the dataset or image at filename
        :rtype: int
        """
        n = self._region_counts[filename]
        if self.filtration is not None:
            n -= self._region_discounts[filename]
        return n

    def get_region_labels_as_list(self) -> List[Any]:
        """
        get_region_labels_as_list returns an ordered list of region labels corresponding to the order of regions in the dataset

        :return: a list of labels
        :rtype: list[Any]
        """
        region_labels = []
        for filename in self._filepaths:
            label = self.get_label(filename)
            regions_in_filename = self.number_of_regions(filename)
            region_labels.extend([label] * regions_in_filename)
        return region_labels

    def __iter__(self):
        i = 0
        while i < len(self):
            yield self[i]
            i += 1
