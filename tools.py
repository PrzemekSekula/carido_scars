"""
This module contains the functions for loading the configuration from a YAML file.
Autor: Przemek Sekula
Created: 2025-10-04
Last modified: 2025-10-04
"""

import yaml

def load_config(config_path="config.yaml"):
    """Loads the configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    # Convert image_size list to tuple
    config['image_size'] = tuple(config['image_size'])
    return config
