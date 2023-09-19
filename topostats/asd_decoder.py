"""For decoding and loading .asd AFM file format into Python Numpy arrays"""
import struct
from typing import BinaryIO

import numpy as np


def read_uint8(open_file: BinaryIO) -> int:
    """Read an unsigned 8 bit integer from an open binary file.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.

    Returns
    -------
    int
        Integer decoded value.
    """
    return int.from_bytes(open_file.read(1), byteorder="little")


def read_int8(open_file: BinaryIO) -> int:
    """Read a signed 8 bit integer from an open binary file.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.

    Returns
    -------
    int
        Integer decoded value.
    """
    return struct.unpack("b", open_file.read(1))[0]


def read_int16(open_file: BinaryIO) -> int:
    """Read a signed 16 bit integer from an open binary file.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.

    Returns
    -------
    int
        Integer decoded value.
    """
    return struct.unpack("h", open_file.read(2))[0]


def read_int32(open_file: BinaryIO) -> int:
    """Read a signed 32 bit integer from an open binary file.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.

    Returns
    -------
    int
        Integer decoded value.
    """
    return struct.unpack("i", open_file.read(4))[0]


def read_uint32(open_file: BinaryIO) -> int:
    """Read an unsigned 32 bit integer from an open binary file.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.

    Returns
    -------
    int
        Integer decoded value.
    """
    return struct.unpack("<I", open_file.read(4))[0]


def read_hex_u32(open_file: BinaryIO) -> str:
    """Read a hex encoded unsigned 32 bit integer value from an open binary file.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.

    Returns
    -------
    str
        String representing a hexadecimal encoded integer value.
    """
    return hex(struct.unpack("<I", open_file.read(4))[0])


def read_float(open_file: BinaryIO) -> float:
    """Read a float from an open binary file.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.

    Returns
    -------
    float
        Float decoded value."""
    return struct.unpack("f", open_file.read(4))[0]


def read_bool(open_file: BinaryIO) -> bool:
    """Read a boolean from an open binary file

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.

    Returns
    -------
    bool
        Boolean decoded value
    """
    return bool(int.from_bytes(open_file.read(1), byteorder="little"))


def read_ascii(open_file: BinaryIO, length_bytes: int = 1) -> str:
    """Read an ascii string of defined length from an open binary file.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.
    length_bytes: int
        Length of the ascii string in bytes that should be read. Default: 1 byte (1 character).

    Returns
    -------
    str
        Ascii text decoded from file.
    """
    return open_file.read(length_bytes).decode("ascii")


def read_null_separated_utf8(open_file: BinaryIO, length_bytes: int = 2) -> str:
    """Read an ascii string of defined length from an open binary file where each
    character is separated by a null byte. This encodingis known as UTF-16LE (Little Endian).
    Eg: b'\x74\x00\x6f\x00\x70\x00\x6f' would decode to 'topo' in this format.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.
    length_bytes: int
        Length of the ascii string in bytes that should be read. Default: 2 bytes
        (1 UTF-16LE character)

    Returns
    -------
    str
        Ascii text decoded from file.
    """
    return open_file.read(length_bytes).replace(b"\x00", b"").decode("ascii")


def skip_bytes(open_file: BinaryIO, length_bytes: int = 1) -> bytes:
    """Skip a specified number of bytes when reading an open binary file.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.
    length_bytes: int
        Number of bytes to skip.

    Returns
    -------
    bytes
        The bytes that were skipped.
    """
    return open_file.read(length_bytes)


def read_header(open_file: BinaryIO) -> dict:
    """Read the header metadata for a .asd AFM file.

    Parameters
    ----------
    open_file: BinaryIO
        An open binary file object.

    Returns
    -------
    header_dict: dict
        Dictionary of header metadata for the .asd file.
    """
    header_dict = {}

    # First 4 bytes are an unsigned integer 32. Presumed to be the file version.
    header_dict["file_version"] = read_int32(open_file)
    # There only ever seem to be two channels available
    # Channel encoding are all in LITTLE ENDIAN format.
    # topology: 0x5054 decodes to 'TP' in ascii little endian
    # error: 0x5245 decodes to 'ER' in ascii little endian
    # phase: 0x4850 decodes to 'PH' in ascii little endian
    header_dict["channel1"] = read_ascii(open_file, 2)
    header_dict["channel2"] = read_ascii(open_file, 2)
    # length of header in bytes - presumably so we can skip it to get to the data
    header_dict["header_length"] = read_int32(open_file)
    # Frame header is the length of the header for each frame to be skipped before reading frame data.
    header_dict["frame_header_length"] = read_int32(open_file)
    header_dict["operator_name_size"] = read_int32(open_file)
    header_dict["comment_offset_size"] = read_int32(open_file)
    header_dict["comment_size"] = read_int32(open_file)
    # x and y resolution (pixels)
    header_dict["x_pixels"] = read_int16(open_file)
    header_dict["y_pixels"] = read_int16(open_file)
    # x and y resolution (nm)
    header_dict["x_nm"] = read_int16(open_file)
    header_dict["y_nm"] = read_int16(open_file)
    # frame time
    header_dict["frame_rate"] = read_float(open_file)
    # z piezo extension
    header_dict["z_piezo_extension"] = read_float(open_file)
    # z piezo gain
    header_dict["z_piezo_gain"] = read_float(open_file)
    # Range of analogue voltage values (for conversion to digital)
    header_dict["analogue_digital_range"] = read_hex_u32(open_file)
    # Number of bits of data for analogue voltage values (for conversion to digital)
    # aka the resolution of the instrument. Usually 12 bits, so 4096 sensitivity levels
    header_dict["analogue_digital_data_bits_size"] = read_int32(open_file)
    header_dict["analogue_digital_resolution"] = 2 ^ header_dict["analogue_digital_data_bits_size"]
    # Not sure, something to do with data averaging
    header_dict["is_averaged"] = read_bool(open_file)
    # Window for averaging the data
    header_dict["averaging_window"] = read_int32(open_file)
    # Some padding to ensure backwards compatilibilty I think
    _ = read_int16(open_file)
    # Date of creation
    header_dict["year"] = read_int16(open_file)
    header_dict["month"] = read_uint8(open_file)
    header_dict["day"] = read_uint8(open_file)
    header_dict["hour"] = read_uint8(open_file)
    header_dict["minute"] = read_uint8(open_file)
    header_dict["second"] = read_uint8(open_file)
    # Rounding degree?
    header_dict["rounding_degree"] = read_uint8(open_file)
    # Maximum x and y scanning range in real space, nm
    header_dict["max_x_scan_range"] = read_float(open_file)
    header_dict["max_y_scan_range"] = read_float(open_file)
    # No idea
    _ = read_int32(open_file)
    _ = read_int32(open_file)
    _ = read_int32(open_file)
    # Number of frames the file had when recorded
    header_dict["initial_frames"] = read_int32(open_file)
    # Actual number of frames
    header_dict["num_frames"] = read_int32(open_file)
    # ID of the AFM instrument
    header_dict["afm_ID"] = read_int32(open_file)
    # ID of the file
    header_dict["file_id"] = read_int16(open_file)
    # Name of the user
    header_dict["user_name"] = read_null_separated_utf8(open_file, length_bytes=header_dict["operator_name_size"])
    # Sensitivity of the scanner in nm / V
    header_dict["scanner_sensitivity"] = read_float(open_file)
    # Phase sensitivity
    header_dict["phase_sensitivity"] = read_float(open_file)
    # Direction of the scan
    header_dict["scan_direction"] = read_int32(open_file)
    # Skip bytes: comment offset size
    _ = skip_bytes(open_file, header_dict["comment_offset_size"])
    # Read a comment
    comment = []
    for _ in range(header_dict["comment_size"]):
        comment.append(chr(read_int8(open_file)))
    header_dict["comment_without_null"] = "".join([c for c in comment if c != "\x00"])

    return header_dict


class VoltageLevelConverter:
    """Class for the conversion of arbitrary level in .asd files to voltage, representing the real
    nanometre heights of the pixels in an image.
    """

    def __init__(self, analogue_digital_range, max_voltage, scaling_factor, resolution):
        self.ad_range = int(analogue_digital_range, 16)
        self.max_voltage = max_voltage
        self.scaling_factor = scaling_factor
        self.resolution = resolution
        print(
            f"created voltage converter. ad_range: {analogue_digital_range} -> {self.ad_range},\
            max voltage: {max_voltage}, scaling factor: {scaling_factor}, resolution: {resolution}"
        )


class UnipolarConverter(VoltageLevelConverter):
    def level_to_voltage(self, level):
        return (self.ad_range * level / self.resolution) * self.scaling_factor


class BipolarConverter(VoltageLevelConverter):
    def level_to_voltage(self, level):
        return (self.ad_range - 2 * level * self.ad_range / self.resolution) * self.scaling_factor
