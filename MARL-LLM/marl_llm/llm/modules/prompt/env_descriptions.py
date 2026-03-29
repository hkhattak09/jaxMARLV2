import yaml
import os

# Environment descriptions for different multi-agent tasks
envs = {
    "assembly": """There are a group of robots and a target region on a 2D plane. The target region is an arbitrarily shaped, simply connected area. 
                    It is discretized into grids, divided by lines parallel to the x and y axes, meaning the target region is composed of a collection of cells.
                    Each robot has a collision radius and a sensing radius. A collision occurs when the distance between two robots is less than twice the collision radius.
                    Additionally, a cell is considered occupied by a robot if the distance between the robot and the cell is less than the collision radius.""".replace("\n                    ", " ")
}

def get_env_description(task_name: str | list = None) -> list[str]:
    """
    Get environment descriptions for specified tasks.
    
    Retrieves task descriptions that define the multi-agent environment
    setup, robot capabilities, and objective constraints.
    
    Args:
        task_name (str|list, optional): Task name(s) to retrieve descriptions for.
            When None, returns all available descriptions.
    
    Returns:
        list[str]: List of environment descriptions for requested tasks.
    """
    if task_name is None:
        return list(envs.values())
    elif isinstance(task_name, str):
        return [envs[task_name]]
    elif isinstance(task_name, list):
        return [envs[task] for task in task_name]

def get_envs() -> list[str]:
    """
    Get all available environment/task names.
    
    Returns:
        list[str]: List of available task names that can be used
                  with get_env_description().
    """
    return list(envs.keys())

# Load task configuration from YAML file
script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_file_path = os.path.join(script_dir, "../../config/", "experiment_config.yaml")

try:
    with open(yaml_file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    
    # Extract task name from configuration
    task_name = data["arguments"]["--run_experiment_name"]["default"][0]
    
    # Get environment description for the configured task
    ENV_DES = get_env_description(task_name)[0]
    
except FileNotFoundError:
    print(f"Warning: Configuration file not found at {yaml_file_path}")
    ENV_DES = ""
except KeyError as e:
    print(f"Warning: Missing key in configuration file: {e}")
    ENV_DES = ""