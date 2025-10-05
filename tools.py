"""
Various tools
Created by: Przemek Sekula
Jan 25, 2025
"""

import os
import sys
import argparse
import pathlib
import yaml
from types import SimpleNamespace

# Optional helper to map Python values to argument types. You can tweak
# this as needed.


def guess_type(value):
    """
    Return a function or type that can parse the given value from a string.
    For example, if 'value' is bool, return a function that interprets
    the CLI string as True/False.
    Args:
        value: The value to guess the type for.
    Returns:
        A function or type that can parse the given value from a string.
    """
    if isinstance(value, bool):
        def str2bool(v):
            if isinstance(v, bool):
                return v
            if v.lower() in ("true", "1", "yes", "y"):
                return True
            if v.lower() in ("false", "0", "no", "n"):
                return False
            raise argparse.ArgumentTypeError("Boolean value expected.")
        return str2bool

    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return float

    return str


def parse_with_config_file(parser, config_file=None, defaults_name="defaults"):
    """
    Parse arguments from a config file.
    Args:
        parser: argparse.ArgumentParser instance
        config_file: Path to the config file
        defaults_name: Name of the section in the config file to use as defaults
    Returns:
        argparse.Namespace: Parsed arguments
    """
    def recursive_update(base, update):
        """
        Recursively update a dictionary 'base' with values from 'update'.
        If 'update' contains nested dicts, they will be merged into 'base' deeply.
        """
        for key, value in update.items():
            if isinstance(
                    value,
                    dict) and key in base and isinstance(
                    base[key],
                    dict):
                recursive_update(base[key], value)
            else:
                base[key] = value

    args, remaining = parser.parse_known_args()

    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
        )

    name_list = [
        defaults_name,
        *args.configs] if args.configs else [defaults_name]

    defaults = {}
    for name in name_list:
        if name not in configs:
            raise ValueError(f"Config '{name}' not found in {config_file}")
        recursive_update(defaults, configs[name])

    parser = argparse.ArgumentParser(description='')
    for key, value in sorted(defaults.items()):
        parser.add_argument(f"--{key}", default=value, type=guess_type(value))

    return parser.parse_args(remaining)

def read_configs(config_file, base_config, config=None):
    """
    Reads configuration from a YAML file, starting with a base configuration,
    and then optionally merging in overrides from another configuration key.
    
    Parameters:
      config_file (str): Path to the YAML configuration file.
      base_config (str): The key name for the base configuration (e.g. "defaults_base").
      config (str, optional): An additional configuration key to merge (e.g. "haulk9").
    
    Returns:
      dict: A dictionary of the resulting configuration parameters.
    
    Raises:
      ValueError: If the base_config or provided config keys are not found.
    """
    # Load the YAML file
    with open(config_file, 'r') as f:
        configs = yaml.safe_load(f)
    
    # Ensure the base configuration exists
    if base_config not in configs:
        raise ValueError(f"Base config '{base_config}' not found in {config_file}.")
    
    # Start with the base configuration
    final_config = configs[base_config].copy()
    
    # If an additional config is provided, update the final configuration
    if config is not None:
        if config not in configs:
            raise ValueError(f"Config '{config}' not found in {config_file}.")
        final_config.update(configs[config])

    return SimpleNamespace(**final_config)

def log_args_to_tensorboard(logger, args):
    """
    Logs all the argparse parameters to TensorBoard.
    
    Args:
        logger (TensorBoardLogger): The TensorBoard logger instance.
        args (Namespace): The argparse Namespace containing the parameters.
    """
    # Convert args to a dictionary of hyperparameters.
    hparams = vars(args)
    
    # Log hyperparameters using the logger's built-in method.
    logger.log_hyperparams(hparams)
    
    # Optionally, also log a formatted string version.
    formatted_hparams = "\n".join([f"{key}: {value}" for key, value in hparams.items()])
    logger.experiment.add_text("Hyperparameters", formatted_hparams, global_step=0)