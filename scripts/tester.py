  
import torch, progressbar
import torch.nn as nn
import numpy as np

from network import network
from data import dataset
import cv2, json
import matplotlib.pyplot as plt

plt.style.use("ggplot")

model_protocol = {"cvae": network.CVAE, "cnn": network.CNN}
dataset_protocol = {"cvae": dataset.cvae_dataset, "cnn": dataset.cnn_dataset}


class Tester:
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
            "Testing phase [",
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
            max_value=config.data.batch_size, widgets=widgets, term_width=100
        )

        ### logging
        self.logger = open("{}/{}.json".format(config.base_dir, "test"), "w")

        ### model
        self.model = model_protocol[config.model.protocol](config)
        self.load_checkpoints()
        if config.framework.num_gpu > 0:
            self.model.to(device=0)
        self.model.eval()


    def load_checkpoints(self):
        sd = torch.load(
            "{}/{}.pt".format(self.cfg.checkpoint_dir, self.cfg.checkpoint_file),
            map_location=torch.device("cpu"),
        )
        self.model.load_state_dict(sd["parameters"])

    def run(self):
        raise NotImplementedError

class Tester_cvae(Tester):
    def __init__(self, configs):
        super(Tester_cvae, self).__init__(configs)

    def BCE_loss(self,recon_x, x):

        BCE = nn.functional.binary_cross_entropy(
            recon_x, x, reduction='sum')

        return BCE / x.size(0)

    def run(self):
        recon_losses = []
        for idx, (occupancy, depth, state) in enumerate(self.dataloader):
            
            if self.cfg.framework.num_gpu > 0:
                occupancy, depth, state = (
                    occupancy.to(device=0),
                    depth.to(device=0),
                    state.to(device=0),
                )

            # forward
            with torch.no_grad():
                recon_state = self.model.inference(occupancy, depth)
            
            # loss
            l = self.BCE_loss(recon_state, state)


            ## save occupancy, state, reconstructed state for visualization later
            
            recon_losses.append(l.detach().cpu().numpy())

           
        print("finish!")
        print("state reconstruction BCE loss: {:.3f}±{:.3f}".format(np.mean(recon_losses), np.std(recon_losses)))
        json.dump(
            {
                "recon_BCE_Loss": np.array(recon_losses).tolist(),
            },
            self.logger,
        )
        self.logger.close()
