"""Topostats"""
from pkg_resources import get_distribution, DistributionNotFound

from topostats.logs.logs import setup_logger

LOGGER = setup_logger()

# try:
#     print(__name__)
# │   DIST_NAME = __name__
# │   __version__ = get_distribution(DIST_NAME).version
# except DistributionNotFound:
# │   __version__ = 'unknown'
# finally:
# │   del get_distribution, DistributionNotFound
