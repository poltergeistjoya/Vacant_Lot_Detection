from pydantic import BaseModel, ValidationError
import yaml 
from pathlib import Path
from logger import get_logger

log = get_logger()

class Data(BaseModel):
    DIR: str
    NYC_MAPPLUTO: Path | None = None
class EarthEngineConfig(BaseModel):
    PROJECT_ID: str

class SensorNormalization(BaseModel):
    NAIP: float = 255

class Config(BaseModel):
    DATA: Data
    EARTH_ENGINE: EarthEngineConfig
    SENSOR_NORMALIZATION: SensorNormalization | None = None

def load_config(path: str | Path) -> Config:
    """
    Load and validate YAML config file against Config model
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    try:
        yaml_data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML: {e}") from e
    try:
        log.info("YAML loaded")
        return Config(**yaml_data)
    except ValidationError as e:
        raise ValueError(f"Invalid config:\n{e}") from e
    
