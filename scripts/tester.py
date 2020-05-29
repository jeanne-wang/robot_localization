  
import torch, progressbar
import torch.nn as nn
import numpy as np

from network import network
from data import dataset
import cv2, json
import matplotlib.pyplot as plt
from utils.map_utils import Visualizer, Map
import glob, os, os.path as osp
import math
from utils.utils import fig2img

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
        for idx, (occupancy, depth, state, occu_map_path_batch, W_world, H_world) in enumerate(self.dataloader):
            
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

            if self.cfg.vis:
                # get world width and height
                W_world, H_world = W_world.detach().cpu().numpy(), H_world.detach().cpu().numpy()

                # create the desired dir to store the visualization images
                vis_dir = "{}/vis".format(self.cfg.base_dir)
                if not osp.isdir(vis_dir):
                    os.makedirs(vis_dir)

                # do it
                for idy in range(len(state)):
                    # get occu map
                    occu_map_path = occu_map_path_batch[idy]
                    if self.dataset.crop_size == 0:
                        m = Map(osp.join(occu_map_path, 'floorplan.yaml'),
                                laser_max_range=self.dataset.laser_max_range,
                                downsample_factor=self.dataset.downsample_factor)
                    else:
                        m = Map(osp.join(occu_map_path, 'floorplan.yaml'),
                                laser_max_range=self.dataset.laser_max_range,
                                downsample_factor=self.dataset.downsample_factor,
                                crop_size=self.dataset.crop_size)

                    # allocate visualizer object with matplotlib and occu map
                    fig, ax = plt.subplots()
                    vis = Visualizer(m, ax)

                    # draw the essential map
                    vis.draw_map()

                    # recover depth information and draw it
                    depth_xy = depth[idy].view(-1, 2).detach().cpu().numpy()
                    depth_xy[:, 0] = depth_xy[:, 0] * W_world[idy]
                    depth_xy[:, 1] = depth_xy[:, 1] * H_world[idy]
                    vis.draw_obstacles(depth_xy, markeredgewidth=1.5)

                    # recover ground truth position and heading and draw it with RED color
                    input_state = state[idy].detach().cpu().numpy()
                    world_pos = np.array([input_state[0] * W_world[idy], input_state[1] * H_world[idy]])
                    heading = np.rad2deg(input_state[2] * (2*math.pi))
                    vis.drwa_location(world_pos, heading, 'r', markersize=7)

                    # map the reconstructed results and draw it with GREEN color
                    output_state = recon_state[idy].detach().cpu().numpy()
                    world_pos = np.array([output_state[0] * W_world[idy], output_state[1] * H_world[idy]])
                    heading = np.rad2deg(output_state[2] * (2*math.pi))
                    vis.drwa_location(world_pos, heading, 'g', markersize=7)

                    # convert the matplotlib object to PIL.Image object and save it to the desired dir
                    img = fig2img(fig)
                    img.save("{}/{:03d}.png".format(vis_dir, idx * self.cfg.data.batch_size + idy + 1))

                    ### TODO: have multiple sample for the same data and draw the heatmap accordingly

        print("finish!")
        print("state reconstruction BCE loss: {:.3f}±{:.3f}".format(np.mean(recon_losses), np.std(recon_losses)))
        json.dump(
            {
                "recon_BCE_Loss": np.array(recon_losses).tolist(),
            },
            self.logger,
        )
        self.logger.close()
