import yaml

# ========================
# HELPER FUNCTIONS
# ========================

def load_params(filepath, logger=None):
    """
    Load parameters from YAML configuration files with error handling.
    
    Args:
        filepath (str): Path to the YAML file to load
        logger (optional): ROS logger for error reporting
    
    Returns:
        dict: Loaded parameters, empty dict if loading fails
    """
    try:
        with open(filepath, 'r') as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as exc:
        if logger:
            logger.error(f"YAML parsing error in {filepath}: {exc}")
        else:
            print(f"[ERROR] YAML parsing error in {filepath}: {exc}")
    except FileNotFoundError:
        if logger:
            logger.error(f"YAML file not found: {filepath}")
        else:
            print(f"[ERROR] YAML file not found: {filepath}")
    return {}