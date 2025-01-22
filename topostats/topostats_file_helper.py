"""For helper scripts in handling .topostats files."""

from pathlib import Path
import logging

import h5py
import numpy.typing as npt

from topostats.io import hdf5_to_dict
from topostats.logs.logs import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


class TopoFileHelper:
    """
    Helper class for searching through the data in a .topostats (hdf5) file.

    Parameters
    ----------
    topofile : Path
        Path to the .topostats file.

    Examples
    --------
    This class should be used in a Jupyter Notebook or an interactive Python session.

    Creating a helper object.
    .. code-block:: RST
        from topostats.io import TopoFileHelper

        topofile = "path/to/topostats_file.topostats"
        helper = TopoFileHelper(topofile)


    Find the data you're looking for using H5Glance (only works in Jupyter Notebooks).
    .. code-block:: RST
        from H5Glance import H5Glance

        file = "path/to/topostats_file.topostats"
        H5Glance(file)

    Which prints an interactive explorer of the file structure, eg:

    .. code-block:: RST
        ../tests/resources/process_scan_topostats_file_regtest.topostats
            grain_masks
            grain_trace_data
            height_profiles
            filename [📋]: scalar entries, dtype: ASCII string
            image [📋]: 64 x 64 entries, dtype: float64
            image_original [📋]: 64 x 64 entries, dtype: float64
            pixel_to_nm_scaling [📋]: scalar entries, dtype: float64
            topostats_file_version [📋]: scalar entries, dtype: float64

    Where each entry can be clicked on for more information.

    Finding data in a file.
    .. code-block:: RST
        from topostats.io import TopoFileHelper

        topofile = "path/to/topostats_file.topostats"
        helper = TopoFileHelper(topofile)
        helper.find_data(["ordered_trace_heights", "0"])

    .. code-block:: RST
        >>>    [ Searching for ['ordered_trace_heights', '0'] in ./path/to/topostats_file.topostats ]
        >>>    | [search] No direct match found.
        >>>    | [search] Searching for partial matches.
        >>>    | [search] !! [ 1 Partial matches found] !!
        >>>    | [search] └ grain_trace_data/above/ordered_trace_heights/0
        >>>    └ [End of search]

    Get data from a file.
    .. code-block:: RST
        from topostats.io import TopoFileHelper

        topofile = "path/to/topostats_file.topostats"
        helper = TopoFileHelper(topofile)

        data = helper.get_data("ordered_trace_heights/0")

    .. code-block:: RST
        >>> [ Get data ] Data found at grain_trace_data/above/ordered_trace_heights/0, type: <class 'numpy.ndarray'>

    Get data information
    .. code-block:: RST
        from topostats.io import TopoFileHelper

        topofile = "path/to/topostats_file.topostats"
        helper = TopoFileHelper(topofile)

        helper.data_info("grain_trace_data/above/ordered_trace_heights/0")

    .. code-block:: RST
        >>> [ Info ] Data at grain_trace_data/above/ordered_trace_heights/0 is a numpy array with shape: (95,),
        >>> dtype: float64
    """

    def __init__(self, topofile: Path | str) -> None:
        """
        Initialise the TopoFileHelper object.

        Parameters
        ----------
        topofile : Path | str
            Path to the .topostats file.
        """
        self.topofile: Path = Path(topofile)
        with h5py.File(self.topofile, "r") as f:
            self.data: dict = hdf5_to_dict(open_hdf5_file=f, group_path="/")

    def search_partial_matches(self, data: dict, keys: list, current_path: list | None = None) -> list:
        """
        Find partial matches to the keys in the dictionary.

        Recursively search through nested dictionaries and keep only the paths that match the keys in the correct order,
        allowing gaps between the keys.

        Parameters
        ----------
        data : dict
            The dictionary to search through.
        keys : list
            The list of keys to search for.
        current_path : list, optional
            The current path in the dictionary, by default [].

        Returns
        -------
        list
            A list of paths that match the keys in the correct order.
        """
        if current_path is None:
            # Need to initialise the empty list here and not as a default argument since it is mutable
            current_path = []

        partial_matches = []

        def recursive_partial_search(data, keys, current_path) -> None:
            """
            Recursively find partial matches to the keys in the dictionary.

            Recursive function to search through the dictionary and keep only the paths
            that match the keys in the correct order,
            allowing gaps between the keys.

            Parameters
            ----------
            data : dict
                The dictionary to search through.
            keys : list
                The list of keys to search for.
            current_path : list
                The current path in the dictionary.
            """
            # If have reached the end of the current dictionary, return
            if not keys:
                partial_matches.append(current_path)
                return

            current_key = keys[0]

            if isinstance(data, dict):
                for k, v in data.items():
                    new_path = current_path + [k]
                    try:
                        # Check if the current key can be converted to an integer
                        current_key_int = int(current_key)
                        k_int = int(k)
                        # If the current key and the key in the dictionary can be converted to integers,
                        # check if they are equal
                        if current_key_int == k_int:
                            # If the current key is in the key list of the dictionary, continue searching
                            # but remove the current key from the list
                            remaining_keys = keys[1:]
                            recursive_partial_search(v, remaining_keys, new_path)
                    except ValueError:
                        # If the current key cannot be converted to an integer, allow for partial matches
                        if current_key in k:
                            # If the current key is in the key list of the dictionary, continue searching
                            # but remove the current key from the list
                            remaining_keys = keys[1:]
                            recursive_partial_search(v, remaining_keys, new_path)
                        else:
                            # If the current key is not in the key list of the dictionary, continue searching
                            # but don't remove the current key from the list as it might be deeper in the dictionary
                            recursive_partial_search(v, keys, new_path)

        recursive_partial_search(data, keys, current_path)
        return partial_matches

    def find_data(self, search_keys: list) -> None:
        """
        Find the data in the dictionary that matches the list of keys.

        Parameters
        ----------
        search_keys : list
            The list of keys to search for.
        """
        # Find the best match for the list of keys
        # First check if there is a direct match
        LOGGER.info(f"[ Searching for {search_keys} in {self.topofile} ]")

        try:
            current_data = self.data
            for key in search_keys:
                current_data = current_data[key]

            LOGGER.info("| [search] Direct match found")
        except KeyError:
            LOGGER.info("| [search] No direct match found.")

        # If no direct match is found, try to find a partial match
        LOGGER.info("| [search] Searching for partial matches.")
        partial_matches = self.search_partial_matches(data=self.data, keys=search_keys)
        if partial_matches:
            LOGGER.info(f"| [search] !! [ {len(partial_matches)} Partial matches found] !!")
            for index, match in enumerate(partial_matches):
                match_str = "/".join(match)
                if index == len(partial_matches) - 1:
                    prefix = "| [search] └"
                else:
                    prefix = "| [search] ├"
                LOGGER.info(f"{prefix} {match_str}")
        else:
            LOGGER.info("| [search] No partial matches found.")
        LOGGER.info("└ [End of search]")

    def get_data(self, location: str) -> int | float | str | npt.NDArray | dict | None:
        """
        Retrieve data from the dictionary using a '/' separated string.

        Parameters
        ----------
        location : str
            The location of the data in the dictionary, separated by '/'.

        Returns
        -------
        int | float | str | np.ndarray | dict
            The data at the location.
        """
        # If there's a trailing '/', remove it
        if location[-1] == "/":
            location = location[:-1]
        keys = location.split("/")

        try:
            current_data = self.data
            for key in keys:
                current_data = current_data[key]
            LOGGER.info(f"[ Get data ] Data found at {location}, type: {type(current_data)}")
            return current_data
        except KeyError as e:
            LOGGER.error(f"[ Get data ] Key not found: {e}, please check the location string.")
            return None

    def data_info(self, location: str, verbose: bool = False) -> None:
        """
        Get information about the data at a location.

        Parameters
        ----------
        location : str
            The location of the data in the dictionary, separated by '/'.

        verbose : bool, optional
            Print more detailed information about the data, by default False.
        """
        # If there's a trailing '/', remove it
        if location[-1] == "/":
            location = location[:-1]
        keys = location.split("/")

        try:
            current_data = self.data
            for key in keys:
                current_data = current_data[key]
        except KeyError as e:
            LOGGER.error(f"[ Info ] Key not found: {e}, please check the location string.")
            return

        if isinstance(current_data, dict):
            key_types = {type(k) for k in current_data.keys()}
            value_types = {type(v) for v in current_data.values()}
            LOGGER.info(
                f"[ Info ] Data at {location} is a dictionary with {len(current_data)} "
                f"keys of types {key_types} and values "
                f"of types {value_types}"
            )
            if verbose:
                for k, v in current_data.items():
                    LOGGER.info(f"  {k}: {type(v)}")
        elif isinstance(current_data, npt.NDArray):
            LOGGER.info(
                f"[ Info ] Data at {location} is a numpy array with shape: {current_data.shape}, "
                f"dtype: {current_data.dtype}"
            )
        else:
            LOGGER.info(f"[ Info ] Data at {location} is {type(current_data)}")

        return
