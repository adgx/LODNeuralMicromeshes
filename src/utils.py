from enum import Enum
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging 

class EColor(Enum):
    RED = '\033[31m'
    BOLD_RED = '\033[1;31m'
    WHITE = '\033[0m'
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    YELLOW = '\033[33m'

class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.INFO: EColor.GREEN.value,
        logging.DEBUG: EColor.BLUE.value,
        logging.WARNING: EColor.YELLOW.value,
        logging.ERROR: EColor.RED.value,
        logging.CRITICAL: EColor.BOLD_RED.value,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, EColor.WHITE.value)
        message = super().format(record)
        return f'{color}{message}{EColor.WHITE.value}'

curr_dir = Path('.')

def save_weights(model, name_weight_file: str):
    assert(Path(curr_dir).exists())
    torch.save(model.state_dict(), curr_dir / name_weight_file)

def load_weights(model, name_weight_file: str):
    model.load_state_dict(torch.load(name_weight_file), wieghts_only=True)

def tensorboard_summary_writer(path:str, name_dir:str) -> SummaryWriter:
    dir = Path(path)
    dir.mkdir(parents=True, exist_ok=True)

    num_dirs = sum(1 for item in dir.iterdir() if item.is_dir())
    global curr_dir
    curr_dir = dir / (name_dir + str(num_dirs))
    return SummaryWriter(str(curr_dir))