import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sci_ndimage
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

def median_flattening(image: np.ndarray, mask: np.ndarray = None):
    if mask is not None:
        read_matrix = np.ma.masked_array(image, mask=mask, fill_value=np.nan).filled()
    else:
        read_matrix = image

    for j in range(image.shape[0]):
        # Get the median of the row
        m = np.nanmedian(read_matrix[j, :])
        # print(m)
        if not np.isnan(m):
            image[j, :] -= m
    return image

def median_differences_flattening(image: np.ndarray, mask: np.ndarray = None):
    if mask is not None:
        read_matrix = np.ma.masked_array(image, mask=mask, fill_value=np.nan).filled()
    else:
        read_matrix = image

    for j in range(read_matrix.shape[0]):
        if j > 0:
            median_height_diff_above = np.nanmedian(read_matrix[j, :] - read_matrix[j-1, :])
            # print(f'median difference in height: {median_height_diff_above}')
            image[j, :] -= median_height_diff_above
    return image

def remove_plane_tilt(image: np.ndarray, mask: np.ndarray = None):
    if mask is not None:
        read_matrix = np.ma.masked_array(image, mask=mask, fill_value=np.nan).filled()
    else:
        read_matrix = image

    # LOBF
    # Calculate medians
    medians_x = [np.nanmedian(read_matrix[:, i]) for i in range(read_matrix.shape[1])]
    medians_y = [np.nanmedian(read_matrix[j, :]) for j in range(read_matrix.shape[0])]

    # Fit linear x
    px = np.polyfit(range(0,len(medians_x)), medians_x, 1)
    print(f'px: {px}')
    py = np.polyfit(range(0,len(medians_y)), medians_y, 1)
    print(f'py: {py}')

    print(f'px[0]: {px[0]}')
    print(f'np.isnan(px[0]): {np.isnan(px[0])}')

    if px[0] != 0 and not np.isnan(px[0]):
        print('removing x plane tilt')
        for j in range(0, image.shape[0]):
            for i in range(0, image.shape[1]):
                image[j, i] -= px[0] * (i)

    if py[0] != 0 and not np.isnan(py[0]):
        print('removing y plane tilt')
        for j in range(0, image.shape[0]):
            for i in range(0, image.shape[1]):
                image[j, i] -= py[0] * (j)
    
    return image

def remove_quadratic(image: np.ndarray, mask: np.ndarray = None):
    if mask is not None:
        read_matrix = np.ma.masked_array(image, mask=mask, fill_value=np.nan).filled()
    else:
        read_matrix = image

    # Calculate medians
    medians_x = [np.nanmedian(read_matrix[:, i]) for i in range(read_matrix.shape[1])]
    # medians_y = [np.median(image[j, :]) for j in range(image.shape[0])]

    # Fit quadratic x
    px = np.polyfit(range(0,len(medians_x)), medians_x, 2)
    # print(f'polyfit: {px}')

    # Plot quadratic fit
    # plt.plot(medians_x, '.', label='x medians')
    # # plt.plot(medians_y, '.', label='y medians')
    # plt.legend()
    # xs = np.array(range(0, len(medians_x)))
    # fitx = np.array(xs**2 * px[0] + xs * px[1] + px[2])
    # plt.plot(range(0, len(medians_x)), fitx)
    # plt.show()

    # Handle divide by zero
    if px[0] != 0 and not np.isnan(px[0]):
        # Remove quadratic x 
        cx = -px[1]/(2*px[0])
        for j in range(0, image.shape[0]):
            for i in range(0, image.shape[1]):
                # image[j, i] -= (px[0] * i**2 + px[1] * (i) + px[2])
                # print(f'value at i={i} : {image[j, i]} | subtracting : px[0] * i**2 = {px[0]} * {i}**2 = {px[0]}*i**2 -> new value = {image[j, i] - px[0] * i**2}')
                image[j, i] -= px[0] * (i-cx)**2

    # -----------------------------------------------------------
    # # Secondary plot of medians after flattening
    # # Calculate medians
    # medians_x = [np.nanmedian(image[:, i]) for i in range(image.shape[1])]
    # # medians_y = [np.median(image[j, :]) for j in range(image.shape[0])]

    # # Fit quadratic x
    # px = np.polyfit(range(0,len(medians_x)), medians_x, 2)
    # # print(px)

    # # Plot quadratic fit
    # plt.plot(medians_x, '.', label='x medians')
    # # plt.plot(medians_y, '.', label='y medians')
    # plt.legend()
    # xs = np.array(range(0, len(medians_x)))
    # fitx = np.array(xs**2 * px[0] + xs * px[1] + px[2])
    # plt.plot(range(0, len(medians_x)), fitx)
    # plt.show()
    # ------------------------------------------------------------

    return image

def add_circles(image: np.ndarray, height: int = 10, number: int = 10, min_size: int = 20, max_size: int = 50, min_thickness: int = 2, max_thickness: int = 10):
    sx = image.shape[1]
    sy = image.shape[0]

    for _ in range(number):
        size = np.random.randint(min_size, max_size)
        thickness = np.random.randint(min_thickness, max_thickness)
        image = add_ring(image, outer_size=size + thickness, inner_size=size, centre=(np.random.randint(0,sx), np.random.randint(0, sy)), height=height)

    return image

def add_ring(image: np.ndarray, outer_size: int = 200, inner_size: int = 100, centre: tuple = (0, 0), height: int = 10):
    cx = centre[0]
    cy = centre[1]
    for i in range(-outer_size, outer_size):
        for j in range(-outer_size, outer_size):
            if j+cy < image.shape[0] - 1:
                if i+cx < image.shape[1] - 1:
                    if ((i)**2 + (j)**2) < outer_size**2 and i**2 + j**2 > inner_size**2:             
                        image[i+cx, j+cy] += height
    return image

def add_scan_lines(image: np.ndarray, magnitude: int = 10):
    for j in range(image.shape[0]):
        image[j, :] += (np.random.random() - 0.5) * magnitude
    return image

def add_random_noise(image: np.ndarray, magnitude: int = 1):
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            image[j, i] += (np.random.random() - 0.5) * magnitude
    return image

def add_slant(image: np.ndarray, magnitude_x: float = 0.01, magnitude_y: float = 0.03):
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            image[j, i] += (magnitude_y*j) + (magnitude_x*i)
    return image

def add_quadratic(image: np.ndarray, a: float = 0, b: float = 0):
    cx = int(image.shape[1]/2)
    _cy = int(image.shape[0]/2)
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            image[j, i] += a*(i-cx)**2 + b*(i-cx) 
    return image

def add_scars(image: np.ndarray, number: int = 10, level: float = 15, min_size: int = None, max_size: int = None):
    sx = image.shape[1]
    sy = image.shape[0]
    if min_size is None:
        min_size = sx / 10
    if max_size is None:
        max_size = sx / 6
    for _ in range(number):
        left = np.random.randint(0, sx-2)
        right = min(sx-1, left + np.random.randint(min_size, max_size))
        image[np.random.randint(0, sy), left:right] = level
    return image

def plot_histogram(image: np.ndarray, title: str = ''):
    plt.figure(figsize=(4, 4), dpi=80)
    plt.hist(image.flatten(), bins=len(np.unique(image)))
    plt.title(title)
    plt.show()
    
def mask_image(image: np.ndarray, std_dev_multiplier: float = 1.0):
    # Calculate threshold value
    mean = np.mean(image)
    std_dev = np.std(image)
    print(f'mean: {mean} std_dev: {std_dev}')
    threshold = mean + std_dev_multiplier * std_dev
    print(f'threshold: {threshold}')

    mask = image > threshold
    # masked_image = np.ma.masked_array(original_image, mask=mask, fill_value=np.nan).filled()

    return mask

def detect_scars(image: np.ndarray):
    kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
    scars = sci_ndimage.convolve(image, weights=kernel)
    scar_mask = scars > 6
    scar_overlay = np.ma.masked_array(scars, mask=scar_mask, fill_value=np.nan).filled()
    plt.imshow(image, vmin=-1, vmax=6)
    plt.imshow(scars, vmin=-1, vmax=6)
    plt.title('scars')
    plt.colorbar()
    plt.show()
    return image

def plot_medians(image: np.ndarray, axis: str = 'x'):
    # # Calculate medians
    if axis == 'x':
        medians = [np.nanmedian(image[:, i]) for i in range(image.shape[1])]
    elif axis == 'y':
        medians_y = [np.median(image[j, :]) for j in range(image.shape[0])]

    # Fit quadratic x
    px = np.polyfit(range(0,len(medians)), medians, 2)
    # print(px)

    # Plot quadratic fit
    plt.plot(medians, '.', label='x medians')
    # plt.plot(medians_y, '.', label='y medians')
    plt.legend()
    xs = np.array(range(0, len(medians)))
    fitx = np.array(xs**2 * px[0] + xs * px[1] + px[2])
    plt.plot(range(0, len(medians)), fitx)
    plt.show()

def flatten(data: np.ndarray, filename: str, save_path: Path, std_dev_multiplier: float):
    image = np.copy(data)
    original_image = np.copy(image)

    if np.isnan(image).any():
        print("ERROR - IMAGE HAS NANs IN IT - AT BEGINNING")

    # Initial flatten
    image = median_flattening(image)
    image = remove_plane_tilt(image)
    image = remove_quadratic(image)
    
    
    mask = mask_image(image, std_dev_multiplier=std_dev_multiplier)
    image_better = np.copy(original_image)

    if np.isnan(image_better).any():
        print("ERROR - IMAGE HAS NANs IN IT - BETWEEN ORIGINAL AND BETTER FLATTENING")

    # print('median flatten')
    image_better = median_flattening(image_better, mask)
    # plt.imshow(image_better)
    # plt.show()
    # plot_medians(image_better)
    # print('plane flatten')

    if np.isnan(image_better).any():
        print("ERROR - IMAGE HAS NANs IN IT - AFTER 2nd MEDIAN FLATTEN")

    image_better = remove_plane_tilt(image_better, mask)
    # plt.imshow(image_better)
    # plt.show()
    # plot_medians(image_better)
    # print('quadratic flatten')

    if np.isnan(image_better).any():
        print("ERROR - IMAGE HAS NANs IN IT - AFTER 2nd PLANE FLATTEN")

    image_better = remove_quadratic(image_better, mask)
    # plt.imshow(image_better)
    # plt.show()
    # plot_medians(image_better)
    # print('median flatten')
    
    if np.isnan(image_better).any():
        print("ERROR - IMAGE HAS NANs IN IT - AFTER 2nd QUADRATIC FLATTEN")
    
    image_better = median_flattening(image_better, mask)
    # plt.imshow(image_better)
    # plt.show()
    # plot_medians(image_better)

    # Plot the images  
    if np.isnan(image_better).any():
        print("ERROR - IMAGE HAS NANs IN IT - AFTER 3rd MEDIAN FLATTEN")

    vmin = -1
    vmax = 3
    fig, ax = plt.subplots(4, 2)
    im1 = ax[0, 0].imshow(original_image, interpolation='None', cmap='afmhot')
    ax[0, 0].set_title('original image')
    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')
    ax[0, 1].hist(original_image.flatten(), bins=len(np.unique(original_image)), log=True)
    ax[0, 1].set_title('original image')

    im2 = ax[1, 0].imshow(image, interpolation='None', vmin=vmin, vmax=vmax, cmap='afmhot')
    ax[1, 0].set_title('flattened image')
    divider = make_axes_locatable(ax[1, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')


    im2 = ax[2, 0].imshow(image_better, interpolation='None', vmin=-3, vmax=3, cmap='afmhot')
    ax[2, 0].set_title('flattened image with mask')
    divider = make_axes_locatable(ax[2, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')

    im2 = ax[3, 0].imshow(mask, interpolation='None')
    ax[3, 0].set_title('mask')
    divider = make_axes_locatable(ax[3, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    medians = [np.nanmedian(image[:, i]) for i in range(image.shape[1])]
    ax[3, 1].plot(medians, '.')
    ax[3, 1].set_title('median distribution x')

    # Histograms
    ax[1, 1].hist(image.flatten(), bins=len(np.unique(original_image)), log=True)
    ax[1, 1].set_title('flattened image')

    print(f'len(np.unique(data)): {len(np.unique(image_better.flatten()))}')
    ax[2, 1].hist(image_better.flatten(), bins=len(np.unique(original_image)), log=True)
    ax[2, 1].set_title('flattened image with mask')


    
    # plt.legend()
    fig.tight_layout()
    fig.suptitle(filename, size=16, y=1.12)
    fig.savefig(f'{Path(save_path) / filename}.png')
    # plt.show()
