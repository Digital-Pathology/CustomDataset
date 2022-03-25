
"""
    filtration_cacheing

        A system for caching information about which regions of an image pass through a filter.
        Includes metadata verification and context management.
        See example use of FiltrationCacheManager
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from multiprocessing import Pool
import os
from typing import Dict, Iterable, Tuple, Union

import tables as pt
import numpy as np

from unified_image_reader import Image

from .util import listdir_recursive
from . import config

# TODO - parameterize region dimensions


class FiltrationCache(AbstractContextManager):

    """
        Tracks images' regions' filtration statuses in a PyTables hdf5 database
    """

    class Description(pt.IsDescription):
        """ the structure for tables in the database """
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
                 h5filepath: Union[str,
                                   None] = config.DEFAULT_FILTRATION_CACHE_FILEPATH,
                 h5filetitle: Union[str,
                                    None] = config.DEFULAT_FILTRATION_CACHE_TITLE,
                 region_dims: tuple = config.REGION_DIMS) -> None:
        """
            FiltrationCache init - opens the h5file which cannot be used otherwise while open here

            h5filepath (str): the path to the file the database should be stored in
                                (default filtration_cache.h5)
            h5filetitle (str): a name for the database (default filtration_cache)
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

    def get_status(self, filtration, filepath, region_index: Union[int, Iterable, None] = None):
        """
            Gets one or all records from table for filtration, os.path.basename(filepath)

            filtration (str): a string representing the filtration applied to the image
            filepath (str): a filepath representing the image filtration was applied to
            region_index_base (int, Iterable(list, tuple, etc.), None): a number representing
                                                                the region of the image filtration
                                                                was applied to (None gets all
                                                                records)
        """
        group = self._get_group(filtration)
        table = self._get_table(group, filepath)
        if region_index is None:  # get all
            return table.read()
        elif isinstance(region_index, Iterable):
            raise NotImplementedError(region_index)
        elif region_index == int(region_index): # handles most library integer types
            return table[region_index]
        else:
            raise TypeError(type(region_index))

    def has_data(self, filtration, filepath) -> bool:
        """ checks if the filtrationcache has a table at filtration/filepath """
        group = self._get_group(filtration, create_if_missing=False)
        if group is None:
            return False
        table = self._get_table(group, filepath, create_if_missing=False)
        if table is None or table.nrows == 0:
            return False
        return True

    def preprocess(self, filtration, filepath, overwrite: bool = True) -> None:
        """
            Applies filtration to image(s) at filepath (listdir recursive)

            filtration: filtration applied to images' regions'
            filepath (str): if a directory, applied to all files in directory
        """
        if os.path.isdir(filepath):  # is directory of image files
            filepaths = listdir_recursive(filepath)
            for f in filepaths:
                self.preprocess(filtration, f)
        else:  # is image file
            print("preprocessing file", filepath)
            group, table = None, None
            if self.has_data(filtration, filepath):
                if not overwrite:
                    return
                else:
                    group = self._get_group(filtration)
                    table = self._get_table(group, filepath)
                    clear_table(table)
                    for key in FiltrationCache.metadata_fields:
                        del table.attrs[key]
            records, dark_regions_total = preprocess(filtration, filepath)
            group = self._get_group(filtration)
            table = self._get_table(group, filepath)
            row = table.row
            for i in range(len(records)):
                row["region_index_base"] = i
                row["region_index_target"] = records[i]["region_index_target"]
                row["filtration_status"] = records[i]["filtration_status"]
                row.append()
            table.attrs["_image_filepath"] = os.path.basename(filepath)
            table.attrs["_image_size"] = Image(filepath).dims
            table.attrs["_image_region_count"] = len(records)
            table.attrs["_image_dark_regions_count"] = dark_regions_total
            table.attrs["_image_regions_discounted"] = len(
                records) - dark_regions_total
            table.attrs["_image_region_dims"] = self.region_dims
            table.flush()

    def get_metadata(self, filtration, filepath) -> Union[int, None]:
        """ the metadata for a table if it exists """
        if not self.has_data(filtration, filepath):
            return None
        else:
            group = self._get_group(filtration)
            table = self._get_table(group, filepath)
            return {f: table.attrs[f] for f in FiltrationCache.metadata_fields}

    def _get_group(self, filtration: str, create_if_missing: bool = True) -> pt.Group:
        """
            returns the group representing the filtration (creates it if needed)

            filtration (str): a string representing the applied filtration
        """
        filtration = str(filtration)
        filtration = preprocess_filtration(filtration)
        group = None
        if filtration not in self.h5file.root:
            if create_if_missing:
                group = self.h5file.create_group("/", filtration, filtration)
        else:
            group = self.h5file.root.__getattr__(filtration)
        return group

    def _get_table(self,
                   group: pt.Group,
                   filepath: str,
                   create_if_missing: bool = True) -> pt.Table:
        """
            returns the table associated with basename(filepath) under group (creates it if needed)

            group (pt.Group): the group the table should belong to
            filepath (str): filepath basename points to an image file.
        """
        filepath = preprocess_filepath(filepath)
        table = None
        if filepath not in group:
            if create_if_missing:
                table = self.h5file.create_table(group, os.path.basename(
                    filepath), FiltrationCache.Description, os.path.basename(filepath))
        else:
            table = group.__getattr__(filepath)
        return table

    def __del__(self):
        """ ensures the file is closed after use - very important! context management is recommended """
        self.h5file.close()

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwagrs):
        self.__del__()


def preprocess(filtration, filepath):
    """ returns a mapping of region index to (fitration status and dark-region-mapped region index) """
    records, dark_regions_total = _apply_filtration_to_regions(filtration, filepath)
    _apply_dark_region_mapping(records, dark_regions_total)
    return records, dark_regions_total


def _apply_filtration_to_regions(filtration, filepath: str) -> Tuple[Dict[int, Dict], int]:
    """ returns a dictionary mapping region_index to filtration_status """
    img = Image(filepath)
    if img.number_of_regions() == 0:
        raise Exception(f"no regions of size 512,512 in {filepath=}")
    records = None
    try:
        pool = Pool()
        # TODO - region dims not parameterized here
        def info_generator():
            for region_index in range(img.number_of_regions()):
                yield (filepath, filtration, region_index)
        records = {i: r for i,r in pool.starmap(process_region, info_generator())}
    finally:
        pool.close()
        print("pool closed, joining")
        pool.join()
    print("finished pool:", len(records), img.number_of_regions())
    if len(records) != img.number_of_regions():
        raise Exception()
    dark_regions_total = 0
    for record in records.values():
        if record["filtration_status"] is False:
            dark_regions_total += 1
    return records, dark_regions_total


def _apply_dark_region_mapping(records: dict, dark_regions_total: int) -> None:
    discounted_size = len(records) - dark_regions_total
    i = 0
    dark_regions_passed = 0
    while i < discounted_size:
        if records[i]["filtration_status"] is False:
            dark_regions_passed += 1
            temp_count = dark_regions_passed
            j = len(records) - 1
            while True:
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
    """ clears a pytables table of all contents (currently leaves metadata) """
    table.remove_rows(0)


def preprocess_filepath(filepath: str):
    """ pytables can't handle certain characters """
    return os.path.basename(filepath).replace('.', '_DOTSYMBOL_')


def postprocess_filepath(filepath: str):
    """ pytables can't handle certain characters """
    return filepath.replace('_DOTSYMBOL_', '.')


def preprocess_filtration(filtration: str):
    """ removes whitespace for pytables compatibility """
    return "".join(filtration.split())

def process_region(filepath, filtration, region_index):
    """ applies filtration to the specified region of the given image """
    return (region_index, {
        "region_index_target": -1,
        "filtration_status": filtration(Image(filepath).get_region(region_index))
    })
