import torch, progressbar, sys, os
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from network import network
from data import dataset

model_protocol = {"cvae": network.CVAE, "cnn": network.CNN}
dataset_protocol = {"cvae": dataset.cvae_dataset, "cnn": dataset.cnn_dataset}


class Trainer:
    def __init__(self, config):

        ### somethings
        self.cfg = config
        self.dataset = dataset_protocol[config.data.protocol](config)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config.data.batch_size,
            num_workers=config.framework.num_thread,
        )
        widgets = [
            "Training phase [",
            progressbar.SimpleProgress(),
            "] [",
            progressbar.Percentage(),
            "] ",
            progressbar.Bar(marker="█"),
            " (",
            progressbar.Timer(),
            " ",
            progressbar.ETA(),
            ") ",
        ]
        self.bar = progressbar.ProgressBar(
            max_value=config.train.num_epoch, widgets=widgets, term_width=100
        )
        self.best_loss = sys.maxsize

        ### logging
        self.logger = SummaryWriter("{}/runs_{}".format(config.base_dir, config.mode))

        ### model
        self.model = model_protocol[config.model.protocol](config)
        if config.framework.num_gpu > 0:
            self.model.to(device=0)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.train.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=config.train.lr_ms, gamma=0.1
        )

    def save_checkpoints(self, losses, avg_period=5):
        if not os.path.isdir(self.cfg.checkpoint_dir):
            os.makedirs(self.cfg.checkpoint_dir)

        sd = {"parameters": self.model.state_dict(), "epoch": len(losses)}
        checkpoint_dir = "{}/{:05d}.pt".format(self.cfg.checkpoint_dir, len(losses))
        torch.save(sd, checkpoint_dir)

        loss = np.mean(losses[-avg_period:])
        if loss < self.best_loss:
            checkpoint_dir = "{}/best_model.pt".format(self.cfg.checkpoint_dir)
            torch.save(sd, checkpoint_dir)
            self.best_loss = loss

    def load_checkpoints(self, config, model):
        sd = torch.load(
            "{}/{}.pt".format(config.checkpoint_dir, config.checkpoint_file),
            map_location=torch.device("cpu"),
        )
        model.load_state_dict(sd["parameters"])

    def run(self):
        raise NotImplementedError


class Trainer_cvae(Trainer):
    def __init__(self, configs):
        super(Trainer_cvae, self).__init__(configs)

    def cvae_loss(self,recon_x, x, mean, log_var):
        BCE = nn.functional.binary_cross_entropy(
            recon_x, x, reduction='sum')
    
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD) / x.size(0)

    def run(self):
        losses = []
        for epoch in range(1, self.cfg.train.num_epoch + 1):
            for idx, (occupancy, depth, state, _, _, _) in enumerate(self.dataloader):
                if self.cfg.framework.num_gpu > 0:
                    occupancy, depth, state = (
                        occupancy.to(device=0),
                        depth.to(device=0),
                        state.to(device=0),
                    )

                # forward
                recon_state, means, log_var, z = self.model(state, occupancy, depth)

                # loss
                l = self.cvae_loss(recon_state, state, means, log_var)

                # backprop
                self.model.zero_grad()
                l.backward()
                self.optimizer.step()

                # log
                self.logger.add_scalar(
                    "{}/loss".format(self.cfg.mode),
                    l.data,
                    idx
                    + (
                        self.dataset.__len__()
                        / self.cfg.data.batch_size
                    )
                    * epoch,
                )
                losses.append(l.detach().cpu().numpy())

            if epoch % self.cfg.train.save_iter == 0:
                self.save_checkpoints(losses)
            self.scheduler.step()
            self.bar.update(epoch)
        print("finish!")
