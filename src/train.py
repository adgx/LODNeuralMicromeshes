import argparse
import math
import numpy as np
import torch
import torch.optim as optim

if __package__:
    from . import config
    from . import utils
    from .utils import EColor
    from .config import logger
    from .tmlp import TMLP, TMLPLoss
else:
    import config
    import utils
    from utils import EColor
    from config import logger
    from tmlp import TMLP, TMLPLoss

class Trainer:
    def __init__(self, 
                 mesh, 
                 tb_path, 
                 hidden_layers, 
                 features, 
                 batch, 
                 lr, 
                 epoches, 
                 device):
        logger.debug("Trainer configuration")
        #dataLoder 
        self.mesh = mesh
        #model
        self.features = features
        self.batch = batch
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self.epoches = epoches
        #TMLP
        self.tmlp = TMLP(num_hidden_layers=hidden_layers)
        self.tmlp.to(self.device)
        self.lr = lr
        #Loss
        logger.debug("loss initailization")
        self.loss_fn = TMLPLoss(hidden_layers + 1, *[(idx + 1) / (hidden_layers + 1) for idx in range(hidden_layers + 1)])
        #optimizer
        self.optimizer = optim.Adam(self.tmlp.parameters(), self.lr)
        #get data datasets
        #fake data
        self.mesh_displacement_map = torch.tensor([1., 2., 3., 4.])
        self.mesh_subdiv = torch.tensor([1, 2, 3, 4])
        self.mesh_points = torch.tensor([[0.1, 0.1, 0.1], 
                                         [0.2, 0.2, 0.2], 
                                         [0.3, 0.3, 0.3], 
                                         [0.4, 0.4, 0.4]])
        assert(self.batch <= len(self.mesh_points))
        #tensorboard
        self.writer = utils.tensorboard_summary_writer(tb_path, config.TENSORBOARD_FILE_NAME)
        self.writer.add_graph(self.tmlp, [self.mesh_points.to(self.device)])

    def train(self) -> None:
        logger.debug("Running the trainig")

        assert(self.mesh_points.shape[0] == self.mesh_subdiv.shape[0])

        batches_per_epoch = (self.mesh_points.shape[0] + self.batch - 1) // self.batch
        eval_frequency = max(1, math.ceil(batches_per_epoch * 0.25))
        tot_steps = self.epoches * batches_per_epoch 
        step = 1

        for idx_ep in range(self.epoches):
            running_loss = 0.0 
            freq_loss = 0.0
            for batch_points, batch_layers, batch_displacement in zip(torch.split(self.mesh_points, self.batch, dim=0), 
                                                                      torch.split(self.mesh_subdiv, self.batch, dim=0), 
                                                                      torch.split(self.mesh_displacement_map, self.batch, dim=0)):

                output = self.tmlp(batch_points.to(self.device))
                loss = self.loss_fn(output, batch_displacement.to(self.device))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                
                if  step % eval_frequency == 0:
                    loss_avg = (running_loss - freq_loss) / eval_frequency

                    epoch = step * self.batch / self.mesh_points.shape[0]
                    logger.debug("step {} / {} (epoch {:.2f} / {}):".format(step, 
                                                                     tot_steps, 
                                                                     epoch, 
                                                                     self.epoches))
                    logger.debug("learning_rate = {:2f}, loss average = {:4f}):".format(self.lr, loss_avg))
                    utils.save_weights(self.tmlp, config.WEIGHTS_FILE_NAME)
                    self.writer.add_scalar('training_loss',
                                           loss_avg * eval_frequency,
                                           step)
                    freq_loss = running_loss
                step += 1
        self.writer.flush()
        self.writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #Data I/O
    #dataset?
    parser.add_argument('--mesh', type=str, help='Target mesh')
    #TMPL
    parser.add_argument('--features', type=int, default=20, help='Feature vector size')
    parser.add_argument('--hidden_layers', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=64, help='size of hidden layer')
    parser.add_argument('--batch', type=int, default=4, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epoches', type=int, default=1, help='number of epoches')
    parser.add_argument('--num_steps', type=int, default=2000, help='number of training steps')
    parser.add_argument('--checkpoint', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    
    #Sampling parameters?
    #Tensorboard summary
    parser.add_argument('--tensorboard_path', type=str, default='./runs', help='path where to store tesorboard data')
    #parser.add_argument('--steps_til_ckpt', type=int, default=50000, help='epoch frequency to save a check point')
    #parser.add_argument('--steps_til_summary', type=int, default=1000, help='epoch frequency to update tensorboard summary')

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

    trainer = Trainer(args.mesh,
                      args.tensorboard_path,
                      args.hidden_layers,
                      args.features, 
                      args.batch, 
                      args.lr,
                      args.epoches, 
                      device)
    trainer.train()