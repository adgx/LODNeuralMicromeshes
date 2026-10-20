import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

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