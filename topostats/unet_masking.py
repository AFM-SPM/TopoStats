"""Segment grains using a U-Net model."""


def make_crop_square(
    crop_min_row: int, crop_min_col: int, crop_max_row: int, crop_max_col: int, image_shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """
    Make a crop square.

    Parameters
    ----------
    crop_min_row : int
        The minimum row index of the crop.
    crop_min_col : int
        The minimum column index of the crop.
    crop_max_row : int
        The maximum row index of the crop.
    crop_max_col : int
        The maximum column index of the crop.
    image_shape : tuple[int, int]
        The shape of the image.

    Returns
    -------
    tuple[int, int, int, int]
        The new crop indices.
    """
    crop_height = crop_max_row - crop_min_row
    crop_width = crop_max_col - crop_min_col

    diff: int
    new_crop_min_row: int
    new_crop_min_col: int
    new_crop_max_row: int
    new_crop_max_col: int

    if crop_height > crop_width:
        # The crop is taller than it is wide
        diff = crop_height - crop_width
        # Check if we can expand equally in each direction
        if crop_min_col - diff // 2 >= 0 and crop_max_col + diff - diff // 2 < image_shape[1]:
            new_crop_min_col = crop_min_col - diff // 2
            new_crop_max_col = crop_max_col + diff - diff // 2
        # If we can't expand uniformly, expand in just one direction
        # Check if we can expand right
        elif crop_max_col + diff - diff // 2 < image_shape[1]:
            # We can expand right
            new_crop_min_col = crop_min_col
            new_crop_max_col = crop_max_col + diff
        elif crop_min_col - diff // 2 >= 0:
            # We can expand left
            new_crop_min_col = crop_min_col - diff
            new_crop_max_col = crop_max_col
        # Set the new crop height to the original crop height since we are just updating the width
        new_crop_min_row = crop_min_row
        new_crop_max_row = crop_max_row
    elif crop_width > crop_height:
        # The crop is wider than it is tall
        diff = crop_width - crop_height
        # Check if we can expand equally in each direction
        if crop_min_row - diff // 2 >= 0 and crop_max_row + diff - diff // 2 < image_shape[0]:
            new_crop_min_row = crop_min_row - diff // 2
            new_crop_max_row = crop_max_row + diff - diff // 2
        # If we can't expand uniformly, expand in just one direction
        # Check if we can expand down
        elif crop_max_row + diff - diff // 2 < image_shape[0]:
            # We can expand down
            new_crop_min_row = crop_min_row
            new_crop_max_row = crop_max_row + diff
        elif crop_min_row - diff // 2 >= 0:
            # We can expand up
            new_crop_min_row = crop_min_row - diff
            new_crop_max_row = crop_max_row
        # Set the new crop width to the original crop width since we are just updating the height
        new_crop_min_col = crop_min_col
        new_crop_max_col = crop_max_col
    else:
        # If the crop is already square, return the original crop
        new_crop_min_row = crop_min_row
        new_crop_min_col = crop_min_col
        new_crop_max_row = crop_max_row
        new_crop_max_col = crop_max_col

    return new_crop_min_row, new_crop_min_col, new_crop_max_row, new_crop_max_col
