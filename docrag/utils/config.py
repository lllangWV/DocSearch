import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from platformdirs import user_config_dir

logger = logging.getLogger(__name__)

load_dotenv()

FILE = Path(__file__).resolve()
UTILS_DIR = FILE.parent
PKG_DIR = UTILS_DIR.parent
DATA_DIR = os.getenv("DATA_DIR")

MODELS_DIR = PKG_DIR / "models"
DEFAULT_CFG = UTILS_DIR / "config.yml"
