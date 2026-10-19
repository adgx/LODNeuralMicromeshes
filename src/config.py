import logging
import utils
from utils import EColor


LOGGING_DATEFMT: str = '%Y/%m/%d %H:%M:%S'
LOGGING_FORMAT:str = '%(asctime)s %(levelname)-8s %(message)s'
WEIGHTS_FILE_NAME:str = 'tmlp_weights.pth'
TENSORBOARD_FILE_NAME:str = 'tmlp_experiment'
#Logger
logger = logging.getLogger('LOD_N_M')
logger.setLevel(logging.DEBUG)
#handler
handler = logging.StreamHandler()
handler.setFormatter(utils.ColoredFormatter(LOGGING_FORMAT))
logger.addHandler(handler)