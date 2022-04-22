import argparse as arg
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Union, Dict
from tqdm import tqdm

from topostats.filters import find_images, process_scan#, hello
from topostats.io import read_yaml
from topostats.logs.logs import setup_logger, LOGGER_NAME

LOGGER = setup_logger(LOGGER_NAME)

def create_parser() -> arg.ArgumentParser:
    """Create a parser for reading options."""
    parser = arg.ArgumentParser(description='Process AFM images. Additional arguments over-ride those in the configuration file.')
    parser.add_argument('-c', '--config_file', dest='config_file', required=True, help='Path to a YAML configuration file.')
    parser.add_argument('-b', '--base_dir',
                        dest='base_dir',
                        type=str,
                        required=False,
                        help='Base directory to scan for images.')
    parser.add_argument('-o', '--output_dir',
                        dest='output_dir',
                        type=str,
                        required=False,
                        help='Output directory to write results to.')
    parser.add_argument('-f', '--file_ext',
                        dest='file_ext',
                        help='File extension to scan for')
    parser.add_argument('-a', '--amplify_level',
                        dest='amplify_level',
                        type=float,
                        required=False,
                        help='Amplify signals by the given factor.')
    parser.add_argument('-m', '--mask',
                        dest='mask',
                        type=bool,
                        required=False,
                        help='Mask the image.')
    parser.add_argument('-q', '--quiet',
                        dest='quiet',
                        type=bool,
                        required=False,
                        help='Toggle verbosity.')
    return parser

def update_config(config: dict, args: arg.Namespace) -> Dict:
    """Update the configuration with any arguments

    Parameters
    ----------
    config: dict
        Dictionary of configuration (typically read from YAML file specified with '-c/--config <filename>')
    args: Namespace
        Command line arguments
    Returns
    -------
    Dict
        Dictionary updated with command arguments.
    """
    args = vars(args)
    config_keys = config.keys()
    for arg_key, arg_value in args.items():
        if arg_key in config_keys and arg_value is not None:
            config[arg_key] = arg_value
    config['base_dir'] = convert_path(config['base_dir'])
    config['output_dir'] = convert_path(config['output_dir'])
    print(f'[update_config] output_dir : {config["output_dir"]}')
    return config

def convert_path(path: Union[str, Path]) -> Path:
    """Ensure path is Path object.

    Parameters
    ----------
    path: Union[str, Path]
        Path to be converted.

    Returns
    -------
    Path
        pathlib Path
    """
    return Path().cwd() if path == './' else Path(path)

def main():
    """Run processing."""

    # Parse command line options, load config and update with command line options
    parser = create_parser()
    args = parser.parse_args()
    config = read_yaml(args.config_file)
    config = update_config(config, args)

    if config['quiet']:
        LOGGER.setLevel('ERROR')

    LOGGER.info(f'Configuration file loaded from    : {args.config_file}')
    LOGGER.info(f'Scanning for images in            : {config["base_dir"]}')
    LOGGER.info(f'Output directory                  : {config["output_dir"]}')
    LOGGER.info(f'Looking for images with extension : {config["file_ext"]}')
    img_files = find_images(config['base_dir'])
    LOGGER.info(f'Images with extension {config["file_ext"]} in {config["base_dir"]} : {len(img_files)}')
    LOGGER.debug(f'Configuration : {config}')


    # For debugging (as Pool makes it hard to find things when they go wrong)
    for x in img_files:
        process_scan(message = 'Lalalala', image=x)
        # process_scan(x,
        #              amplify_level=config['amplify_level'],
        #              channel=config['channel'],
        #              gaussian_size=config['grains']['gaussian_size'],
        #              dx=config['grains']['dx'],
        #              upper_height_threshold_rms_multiplier=config['grains']['upper_height_threshold_rms_multiplier'],
        #              lower_threshold_otsu_multiplier=config['grains']['lower_threshold_otsu_multiplier'],
        #              minimum_grain_size=config['grains']['minimum_grain_size'],
        #              mode=config['grains']['mode'],
        #              background=config['grains']['background'],
        #              output_dir=config['output_dir'],
        #              quiet=config['quiet'])
        print('HELLLLLO?')
        LOGGER.info(f'We made it past, where is the output?')
    # processing_function = partial(process_scan,
    #                               amplify_level=config['amplify_level'],
    #                               channel=config['channel'],
    #                               gaussian_size=config['grains']['gaussian_size'],
    #                               dx=config['grains']['dx'],
    #                               upper_height_threshold_rms_multiplier=config['grains']['upper_height_threshold_rms_multiplier'],
    #                               lower_threshold_otsu_multiplier=config['grains']['lower_threshold_otsu_multiplier'],
    #                               minimum_grain_size=config['grains']['minimum_grain_size'],
    #                               mode=config['grains']['mode'],
    #                               background=config['grains']['background'],
    #                               output_dir=config['output_dir'],
    #                               quiet=config['quiet'])

    # with Pool(processes=config['cores']) as pool:
    #     with tqdm(total=len(img_files), desc=f'Processing images from {config["base_dir"]}, results are under {config["output_dir"]}') as pbar:
    #         for _ in pool.imap_unordered(processing_function, img_files):
    #             pbar.update()


if __name__ == '__main__':
    main()
