
"""
    A system for caching information about which regions of an image pass through a filter.
    Includes metadata verification and context management.
"""

from __future__ import annotations

# stdlib imports
from contextlib import AbstractContextManager
from multiprocessing import Pool
import os
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

# pip imports
import tables as pt
from tqdm import tqdm as loadingbar

# in-house imports
import unified_image_reader

# local imports
from . import util
from . import config


# TODO - This class is too big and should be aggregating an adapter on the pytables functionality
# TODO - finish thread-safety precautions
# TODO - improve filtration string representation system


class FiltrationCache(AbstractContextManager):
    """
    FiltrationCache Tracks images' regions' filtration statuses in a PyTables hdf5 database

    :raises NotImplementedError: when region_index is region coordinates instead
    :raises TypeError: when region_index is of the wrong type
    """

    class Description(pt.IsDescription):
        """
        Description - the structure for tables in the database
        """
        region_index_base = pt.IntCol(pos=0)  # region index
        region_index_target = pt.IntCol(pos=1)  # region it maps to
        # filtration status of region index base
        filtration_status = pt.BoolCol(pos=2)

    metadata_fields = [
        "_image_filepath",
        "_image_size",
        "_image_region_count",
        "_image_dark_regions_count",
        "_image_regions_discounted",
        "_image_region_dims",
    ]

    def __init__(self,
                 h5filepath: Optional[util.FilePath] = config.DEFAULT_FILTRATION_CACHE_FILEPATH,
                 h5filetitle: Optional[str] = config.DEFULAT_FILTRATION_CACHE_TITLE,
                 region_dims: Optional[unified_image_reader.util.RegionDimensions] = config.REGION_DIMS) -> None:
        """
        __init__ initializes FiltrationCache

        :param h5filepath: the path to the file the database should be stored in, defaults to config.DEFAULT_FILTRATION_CACHE_FILEPATH
        :type h5filepath: Optional[util.FilePath], optional
        :param h5filetitle: a name for the database, defaults to config.DEFULAT_FILTRATION_CACHE_TITLE
        :type h5filetitle: Optional[str], optional
        :param region_dims: dimensions of the regions for which to cache filtration statuses, defaults to config.REGION_DIMS
        :type region_dims: Optional[unified_image_reader.util.RegionDimensions], optional
        """
        # save parameters
        self.h5filepath = h5filepath
        self.h5filetitle = h5filetitle
        self.region_dims = region_dims
        # open h5file
        self.h5file_openmode = "w" if not os.path.exists(
            self.h5filepath) else "a"
        self.h5file = pt.open_file(
            self.h5filepath, mode=self.h5file_openmode, title=self.h5filetitle)
        # multithreading lock for h5 file writing
        self._lock = util.ThreadingLock()

    def get_status(self,
                   filtration: Union[util.FiltrationRepr, str],
                   filepath: util.FilePath,
                   region_index: Optional[unified_image_reader.util.RegionIndex] = None) -> util.FiltrationStatus:
        """
        get_status gets one or all records from table for filtration, os.path.basename(filepath)

        :param filtration: a string representing the filtration applied to the image
        :type filtration: Union[util.FiltrationRepr, str]
        :param filepath: a filepath representing the image filtration was applied to
        :type filepath: util.FilePath
        :param region_index: the index of the region in question, defaults to None
        :type region_index: Optional[util.RegionIndex], optional
        :raises NotImplementedError: if region_index is coordinates
        :raises TypeError: if region_index is of the wrong type
        :return: a tuple of (region index, target region index, region index filtration status)
        :rtype: util.FiltrationReprStatus
        """
        group = self._get_group(filtration)
        table = self._get_table(group, filepath)
        if region_index is None:  # get all
            with self._lock as permission:
                return table.read()
        elif isinstance(region_index, Iterable):
            raise NotImplementedError(region_index)
        # handles most library integer types
        elif region_index == int(region_index):
            with self._lock as permission:
                return table[region_index]
        else:
            raise TypeError(f"{type(region_index)=}, {region_index=}")

    def has_data(self, filtration: util.FiltrationRepr, filepath: util.FilePath, **kwargs) -> bool:
        """
        has_data checks if the filtrationcache has a table at filtration/filepath

        :param filtration: the filtration for which statuses are checked
        :type filtration: util.FiltrationRepr
        :param filepath: a key for the image in question
        :type filepath: util.FilePath
        :return: whether the FiltrationCache has the data in question
        :rtype: bool
        """
        group = self._get_group(filtration, create_if_missing=False, **kwargs)
        if group is None:
            return False
        table = self._get_table(group, filepath, create_if_missing=False)
        if table is None or table.nrows == 0:
            return False
        return True

    def preprocess(self, filtration: util.FiltrationRepr, filepath: util.FilePath, loadingbars: bool, overwrite: bool = True, **kwargs) -> None:
        """
        preprocess applies filtration to image(s) at filepath (listdir recursive)

        :param filtration: filtration applied to images' regions'
        :type filtration: util.FiltrationRepr
        :param filepath: if a directory, applied to all files in directory
        :type filepath: util.FilePath
        :param overwrite: whether to overwrite existing data, if applicable, defaults to True
        :type overwrite: bool, optional
        """
        if os.path.isdir(filepath):  # is directory of image files
            filepaths = util.listdir_recursive(filepath)
            for f in filepaths:
                self.preprocess(filtration, f)
        else:  # is image file
            #print("preprocessing file", filepath)
            group, table = None, None
            if self.has_data(filtration, filepath, **kwargs):
                if not overwrite:
                    return
                else:
                    group = self._get_group(filtration, **kwargs)
                    table = self._get_table(group, filepath)
                    with self._lock as permission:
                        clear_table(table)
                        for key in FiltrationCache.metadata_fields:
                            del table.attrs[key]
            records, dark_regions_total = preprocess(
                filtration, filepath, self.region_dims, loadingbars=loadingbars)
            group = self._get_group(filtration, **kwargs)
            table = self._get_table(group, filepath)
            with self._lock as permission:
                row = table.row
                for i in range(len(records)):
                    row["region_index_base"] = i
                    row["region_index_target"] = records[i]["region_index_target"]
                    row["filtration_status"] = records[i]["filtration_status"]
                    row.append()
                table.attrs["_image_filepath"] = os.path.basename(filepath)
                table.attrs["_image_size"] = unified_image_reader.Image(
                    filepath).dims
                table.attrs["_image_region_count"] = len(records)
                table.attrs["_image_dark_regions_count"] = dark_regions_total
                table.attrs["_image_regions_discounted"] = len(
                    records) - dark_regions_total
                table.attrs["_image_region_dims"] = self.region_dims
                table.flush()

    def get_metadata(self, filtration: util.FiltrationRepr, filepath: util.FilePath) -> util.FiltrationCacheMetadata:
        """
        get_metadata gets the metadata for a table if it exists

        :param filtration: the filtration for which to get metadata
        :type filtration: util.FiltrationRepr
        :param filepath: the image for which to get metadata
        :type filepath: util.FilePath
        :return: the metadata in question, if available
        :rtype: util.FiltrationCacheMetadata
        """
        if not self.has_data(filtration, filepath):
            return None
        else:
            group = self._get_group(filtration)
            table = self._get_table(group, filepath)
            with self._lock as permission:
                return {f: table.attrs[f] for f in FiltrationCache.metadata_fields}

    def _get_group(self, filtration: util.FiltrationRepr, create_if_missing: bool = True, **kwargs) -> Optional[pt.Group]:
        """
        _get_group returns the group representing the filtration (creates it if needed)

        :param filtration: the filtration in question
        :type filtration: util.FiltrationRepr
        :param create_if_missing: whether to create the group if it doesn't exist, defaults to True
        :type create_if_missing: bool, optional
        :return: the group in question
        :rtype: Optional[pt.Group]
        """
        filtration = str(filtration)
        filtration = preprocess_filtration(filtration, **kwargs)
        group = None
        with self._lock as permission:
            if filtration not in self.h5file.root:
                if create_if_missing:
                    group = self.h5file.create_group(
                        "/", filtration, filtration)
            else:
                group = self.h5file.root.__getattr__(filtration)
        return group

    def _get_table(self,
                   group: pt.Group,
                   filepath: util.FilePath,
                   create_if_missing: bool = True) -> Optional[pt.Table]:
        """
        _get_table returns the table associated with basename(filepath) under group (creates it if needed)

        :param group: the group the table should belong to
        :type group: pt.Group
        :param filepath: filepath basename points to an image file
        :type filepath: util.FilePath
        :param create_if_missing: whether to create the table if it doesn't exist, defaults to True
        :type create_if_missing: bool, optional
        :return: the table in question
        :rtype: Optional[pt.Table]
        """
        filepath = preprocess_filepath(filepath)
        table = None
        with self._lock as permission:
            if filepath not in group:
                if create_if_missing:
                    table = self.h5file.create_table(group, os.path.basename(
                        filepath), FiltrationCache.Description, os.path.basename(filepath))
            else:
                table = group.__getattr__(filepath)
        return table

    def __del__(self):
        """
        __del__ ensures the file is closed after use - very important! context management is recommended
        """
        self.h5file.close()

    def __exit__(self, *args, **kwagrs):
        """
        __exit__ enforces safe deconstruction on exiting context
        """
        self.__del__()


def preprocess(filtration: Callable,
               filepath: util.FilePath,
               region_dims: unified_image_reader.util.RegionDimensions,
               loadingbars: bool,
               **kwargs) -> Tuple[Dict[int, util.FiltrationStatus], int]:
    """
    preprocess returns a mapping of region index to (fitration status and dark-region-mapped region index)

    :param filtration: the filtration to apply to the image's regions. If callable and not strictly filtration, then a ranked threshold approach is used - see _apply_filtration_to_regions_ranked_threshold
    :type filtration: util.FiltrationRepr
    :param filepath: the filepath where the image in question is found
    :type filepath: util.FilePath
    :param region_dims: the dimensions of the regions to which filtration is applied
    :type region_dims: unified_image_reader.util.RegionDimensions
    :return: filtration status records and dark region count
    :rtype: Tuple[Dict[int, util.FiltrationStatus], int]
    """
    records, dark_regions_total = None, None
    if isinstance(filtration, (str, util.Filter, util.FilterManager)):
        records, dark_regions_total = _apply_filtration_to_regions(
            filtration, filepath, region_dims, loadingbars=loadingbars)
    elif isinstance(filtration, Callable):
        records, dark_regions_total = _apply_filtration_to_regions_ranked_threshold(
            measure=filtration,
            filepath=filepath,
            region_dims=region_dims,
            n_regions=kwargs.get('n_regions', 100),
            threshold=kwargs.get('threshold', None),
            loadingbars=loadingbars
        )
    else:
        raise TypeError(type(filtration))
    _apply_dark_region_mapping(records, dark_regions_total)
    return records, dark_regions_total


def _apply_filtration_to_regions(filtration: util.FiltrationRepr, filepath: util.FilePath, region_dims: unified_image_reader.util.RegionDimensions, loadingbars: bool) -> Tuple[Dict[int, util.FiltrationStatus], int]:
    """
    _apply_filtration_to_regions returns a dictionary mapping region_index to filtration_status

    :param filtration: the filtration to apply to the image's regions
    :type filtration: util.FiltrationRepr
    :param filepath: the filepath where the image in question is found
    :type filepath: util.FilePath
    :param region_dims: the dimensions of the regions to which filtration is applied
    :type region_dims: unified_image_reader.util.RegionDimensions
    :raises Exception: if there are no regions in the image of size region_dims
    :raises Exception: if there was an issue with the gathering filtration status into the records struct
    :return: filtration status records and dark region count
    :rtype: Tuple[Dict[int, util.FiltrationStatus], int]
    """
    img = unified_image_reader.Image(filepath)
    if img.number_of_regions() == 0:
        raise Exception(f"no regions of size {region_dims=} in {filepath=}")
    records = None
    if config.FILTRATION_CACHE_APPLY_FILTRATION_MULTIPROCESSING:
        try:
            pool = Pool()

            def info_generator():
                """
                info_generator is a generator for iterating through the regions

                :yield: the information necessary
                :rtype: int
                """
                for region_index in range(img.number_of_regions(region_dims)):
                    yield (filepath, filtration, region_index, region_dims)
            records = {i: r for i, r in pool.starmap(
                process_region, info_generator())}
        finally:
            pool.close()
            pool.join()
        if len(records) == 0 or len(records) != img.number_of_regions():
            raise Exception(f"{len(records)=}, {img.number_of_regions()=}")
    else:  # no multiprocessing
        records = {}
        iterator = range(img.number_of_regions())
        if loadingbars:
            iterator = loadingbar(iterator, total=img.number_of_regions())
        for region_index in iterator:
            records[region_index] = process_region(
                filepath, filtration, region_index, region_dims)[1]
    dark_regions_total = 0
    for record in records.values():
        if record["filtration_status"] is False:
            dark_regions_total += 1
    return records, dark_regions_total


def _apply_filtration_to_regions_ranked_threshold(measure: Callable, filepath: util.FilePath, region_dims: unified_image_reader.util.RegionDimensions, loadingbars: bool, n_regions: Optional[int] = 100, threshold: Optional[Any] = None) -> Tuple[Dict[int, util.FiltrationStatus], int]:
    """
    apply_filtration_to_regions_ranked_threshold filters out all regions except for the top n_regions with measure(region) >= threshold

    :param measure: the measure of a region
    :type measure: Callable
    :param filepath: the filepath of the image in question
    :type filepath: util.FilePath
    :param region_dims: dimensions of each region
    :type region_dims: unified_image_reader.util.RegionDimensions
    :param n_regions: top n_regions pass through filter, defaults to 100
    :type n_regions: Optional[int], optional
    :param threshold: up to top n_regions pass are admitted, excluding those which are below the threshold, defaults to None
    :type threshold: Optional[Any], optional
    :return: filtration status records and dark region count
    :rtype: Tuple[Dict[int, util.FiltrationStatus], int]
    """
    img = unified_image_reader.Image(filepath)
    if img.number_of_regions() == 0:
        raise Exception(f"no regions of size {region_dims=} in {filepath=}")
    measures = []
    iterator = enumerate(img)
    if loadingbars:
        iterator = loadingbar(iterator, total=len(img))
    for region_num, region in iterator:
        region_measure = measure(region)
        measures.append((region_measure, region_num))
    measures.sort()
    with open('temp.measures', 'w' if os.path.exists('temp.measures') else 'x') as f:
        f.write(str(measures))
    records = {i: {"region_index_target": -1, "filtration_status": False}
               for i in range(img.number_of_regions())}
    dark_regions_total = 0
    for measure, region_index in measures[:n_regions]:
        if threshold is None or measure >= threshold:
            records[region_index]["filtration_status"] = True
            dark_regions_total += 1
    return records, dark_regions_total


def _apply_dark_region_mapping(records: Dict[int, util.FiltrationStatus], dark_regions_total: int) -> None:
    """
    _apply_dark_region_mapping applies the dark region mapping algorithm to the records in-place

    :param records: the records to which dark region is applied
    :type records: Dict[int, util.FiltrationStatus]
    :param dark_regions_total: the total number of dark regions among the records
    :type dark_regions_total: int
    """
    discounted_size = len(records) - dark_regions_total
    i = 0
    dark_regions_passed = 0
    while i < discounted_size:
        if records[i]["filtration_status"] is False:
            dark_regions_passed += 1
            temp_count = dark_regions_passed
            j = len(records) - 1
            if j < 0:
                raise Exception("no records")
            while j >= 0:
                if records[j]["filtration_status"] is True:
                    temp_count -= 1
                    if temp_count == 0:
                        break
                j -= 1
            records[i]["region_index_target"] = j
        else:
            records[i]["region_index_target"] = i
        i += 1


def clear_table(table: pt.Table) -> None:
    """
    clear_table clears a pytables table of all contents (currently leaves metadata)

    :param table: the table to clear
    :type table: pt.Table
    """
    table.remove_rows(0)


def preprocess_filepath(filepath: util.FilePath) -> str:
    """
    preprocess_filepath pytables can't handle certain characters

    :param filepath: the filepath to preprocess
    :type filepath: util.FilePath
    :return: a filepath more coherent to pytables's restrictions
    :rtype: str
    """
    return os.path.basename(filepath).replace('.', '_DOTSYMBOL_')


def postprocess_filepath(filepath: util.FilePath) -> str:  # TODO - new type for this
    """
    postprocess_filepath undoes the preprocessing

    :param filepath: the filepath to un-preprocess
    :type filepath: util.FilePath
    :return: the natural filepath
    :rtype: str
    """
    return filepath.replace('_DOTSYMBOL_', '.')


def preprocess_filtration(filtration: util.FiltrationRepr, **kwargs) -> util.FiltrationRepr:
    """
    preprocess_filtration removes whitespace for pytables compatibility

    :param filtration: the filtration to represent
    :type filtration: util.FiltrationRepr
    :return: a pytables-agreeable filtration representation
    :rtype: util.FiltrationRepr
    """
    s = "".join(str(filtration).split())
    for arg, v in kwargs.items():
        s += f",{arg}={v}"
    return s


def process_region(filepath: util.FilePath, filtration: util.FiltrationRepr, region_index: unified_image_reader.util.RegionIndex, region_dims: util.RegionDimensions) -> Tuple[int, Dict[str: Any]]:
    """
    process_region applies filtration to the specified region of the given image

    :return: the region index and the filtration status
    :rtype: Tuple[int, Dict[str: Any]]
    """
    return (region_index, {
        "region_index_target": -1,
        "filtration_status": filtration(unified_image_reader.Image(filepath).get_region(region_index, region_dims))
    })
