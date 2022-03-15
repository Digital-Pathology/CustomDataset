
"""
    Implements a preprocessor for images' regions' filtration statuses
"""

from typing import Callable, Dict, Tuple

from unified_image_reader import Image

def preprocess(filtration: Callable, filepath: str) -> None:
    """
        Process the filtration for an image and store into cache
    """
    records, dark_regions_total = _apply_filtration_to_regions(filtration, filepath)
    _apply_dark_region_mapping(records, dark_regions_total)
    return records, dark_regions_total

def _apply_filtration_to_regions(filtration, filepath: str) -> Tuple[Dict[int, Dict], int]:
    img = Image(filepath)
    records = {}
    dark_regions_total = 0
    for region_index, region in enumerate(img):
        filtration_status = filtration(region)
        records[region_index] = {
            "region_index_target": -1,
            "filtration_status": filtration_status
        }
        if filtration_status is False:
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
