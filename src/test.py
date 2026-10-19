import argparse
import torch

if __package__:
    from . import config
    from . import utils
    from config import logger
    from .utils import EColor
    from .tmlp import TMLP, TMLPLoss
else:
    import config
    import utils
    from config import logger
    from utils import EColor
    from tmlp import TMLP, TMLPLoss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #Data I/O
    #dataset?
    parser.add_argument('--mesh', type=str, help='Target mesh')
    #TMPL
    parser.add_argument('--features', type=int, default=20, help='Feature vector size')
    parser.add_argument('--hidden_layers', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=64, help='size of hidden layer')
    parser.add_argument('--batch', type=int, default=10, help='Batch size for training')
    parser.add_argument('--checkpoint', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    
    #Tensorboard summary
    parser.add_argument('--tensorboard_path', type=str, default='./runs', help='path where to store tesorboard data')

    args = parser.parse_args()

    args = parser.parse_args()
    logger.info('start')
    logger.debug('Checking the accelerator device')
    device = None
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None:
        logger.warning('No accelerator device')
    elif accelerator.type != 'cuda':
        logger.warning(f'Device type: {accelerator.type}')
    else: 
        logger.info(f'Device type: {accelerator.type}')
        device = 'cuda:'  + str(args.gpu)