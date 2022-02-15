
from __future__ import annotations

from contextlib import AbstractContextManager
import json
import os
from types import TracebackType
from typing import Any, Type, Union

from . import config

class FiltrationCacheManager(dict):
    """ 
        FiltrationCacheManager

            A system for managing multiple images' filtration caches as separate files in a single directory 

            Example Use (Training my_model):

                cache_manager = FiltrationCacheManager("data/filtration_cache", my_filter) \ 
                for region, region_num in Image(image_filename):
                    with cache_manager[image_filename] as cache:
                        if region_num not in cache:
                            cache[region_num] = my_filter.filter(region) \ 
                        if cache[region_num]:
                            my_model.train_with(region)
        
    """
    def __init__(self, dirname: str, filtration: str, region_size: tuple = config.region_size):
        """
            FiltrationCacheManager initializer

                dirname (string): the directory where the FiltrationCacheFile(s) will be saved
                filtration (str): a representation of the filtration applied to the images' regions
                region_size (tuple): the size of the regions being filtered
        """
        self.dirname = dirname
        self.filtration = filtration
        self.region_size = region_size
    def __getitem__(self, filename: str) -> Any:
        return FiltrationCacheFile(
            cache_filepath = self.cache_filepath_for_filename(filename),
            image_filename = filename,
            filtration_string = str(self.filtration),
            region_size = self.region_size
        )
    def __setitem__(self, filename: str) -> None:
        raise Exception("FiltrationCacheManager is read-only --> see example use")
    def cache_filepath_for_filename(self, filename):
        """ returns the filepath of the cache file in the directory """
        if os.path.dirname(filename) != self.dirname:
            filename = os.path.join(self.dirname, os.path.basename(filename))
        return filename + ".cache.json"

class FiltrationCacheFile(AbstractContextManager):
    """ A wrapper on FiltrationCache for context management and recording filtration metadata """
    def __init__(self, cache_filepath: str, image_filename: str, filtration_string: str, region_size: tuple = config.region_size) -> None:
        """
            FiltrationCacheFile initializer

                cache_filepath (str): the location of the cache file
                image_filename (str): the name of the image for which region filtration is cached
                filtration_string (str): a representation of the filtration applied to the regions
                region_size (tuple): the size of the regions referenced by the cache

                context_cache (FiltrationCache, None): helps keep track of cache during context management
        """
        self.cache_filepath = cache_filepath
        self.image_filename = image_filename
        self.filtration_string = filtration_string
        self.region_size = region_size
        self.context_cache: Union[FiltrationCache, None] = None
    def __enter__(self) -> FiltrationCache:
        """
            Context Manager start

                First checks that no other contexts have been opened with this object.
                Then, if the file does not exist, creates it with an empty cache and returns the empty cache.
                If the file does exist, reads the file, ensures no metadata conflicts, and returns the contained cache.
        """
        if self.context_cache is not None: raise MultipleContextsException()
        if not os.path.isfile(self.cache_filepath):
            self.context_cache = FiltrationCache()
            self.save()
        with open(self.cache_filepath) as f:
            data = json.load(f)
            self.compare_filenames(self.image_filename, data.get("filename"))
            self.compare_filtrations(self.filtration_string, data.get("filtration"))
            self.compare_region_sizes(self.region_size, data.get("region_size"))
            self.context_cache = FiltrationCache(data.get("cache"))
        return self.context_cache
    def __exit__(self, __exc_type: Union[Type[BaseException], None], __exc_value: Union[BaseException, None], __traceback: Union[TracebackType, None]) -> Union[bool, None]:
        """
            Context Manager end

                First checks that no other contexts have been opened with this object.
                Then saves the contents of context_cache to the file with metadata.
        """
        if self.context_cache is None: raise MultipleContextsException()
        self.save()
        self.context_cache = None
    def save(self):
        data = {
            "filename": self.image_filename,
            "filtration": self.filtration_string,
            "region_size": self.region_size,
            "cache": self.context_cache.cache
        }
        with open(self.cache_filepath, "w" if os.path.isfile(self.cache_filepath) else "x") as f:
            json.dump(data, f)
    def compare_filenames(self, *args):
        if False in [args[0]==fn for fn in args[1:]]:
            raise ConflictingMetadataException(args)
    def compare_filtrations(self, *args):
        if False in [args[0]==fn for fn in args[1:]]:
            raise ConflictingMetadataException(args)
    def compare_region_sizes(self, *args):
        if False in [tuple(args[0])==tuple(rn) for rn in args[1:]]:
            raise ConflictingMetadataException(args)

class FiltrationCache:
    """ A simple wrapper of dict for keeping track of region filtration status """
    def __init__(self, cache: Union[dict, None] = None, default_filtration_result: Union[bool, None] = config.default_filtration_result):
        """ 
            FiltrationCache initializer
                cache (dict, None): optional parameter representing an existing cache
                default_filtration_result (bool): for when a region isn't present in the cache
        """
        if isinstance(cache, dict):
            self.cache = cache
        else:
            self.cache = {}
        self.default_filtration_result = default_filtration_result
    def __getitem__(self, region: int) -> bool:
        return self.cache.get(region, self.default_filtration_result)
    def __setitem__(self, region: int, filter: bool) -> None:
        self.cache[region] = filter

class MultipleContextsException(Exception): pass
class ConflictingMetadataException(Exception): pass
