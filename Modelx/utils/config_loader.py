from addict import Dict
import yaml
import os

def load_config(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file {file_path} not found")
    with open(file_path, 'r') as f:
        return Dict(yaml.safe_load(f))

def get_configs():
    model_config = load_config('configs/model_config.yaml')
    training_config = load_config('configs/training_config.yaml')
    data_config = load_config('configs/data_config.yaml')
    
    # Handle flash_attention (optional)
    flash_config = Dict({'enabled': False})  # Default if not provided
    try:
        flash_config = load_config('configs/flash_attention.yaml')
    except FileNotFoundError:
        print("flash_attention.yaml not found, using default (disabled)")

    return Dict({
        'model': model_config,
        'training': training_config,
        'data': data_config,
        'flash_attention': flash_config
    })