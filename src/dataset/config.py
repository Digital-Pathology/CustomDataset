
import logging as log

LOG_LEVEL = log.DEBUG

log.basicConfig(
    filename = "CustomDataset.log",
    encoding = "utf-8",
    level = LOG_LEVEL
)

REGION_DIMS = (512, 512)

DEFAULT_FILTRATION_STATUS = None

DEFAULT_FILTRATION_CACHE_FILEPATH = "filtration_cache.h5"
DEFULAT_FILTRATION_CACHE_TITLE = "filtration_cache"

LABEL_MANAGER_IF_NO_LABEL = True


