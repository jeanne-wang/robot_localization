  
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

model_protocol = {"cvae": network.CVAE,
                  "cnn": network.CNN,
                  "cls": network.Classfication_model,
                  "reg": network.Regression_model}
dataset_protocol = {"cvae": dataset.cvae_dataset,
                    "cnn": dataset.cnn_dataset,
                    "cls": dataset.cvae_dataset,
                    "reg": dataset.cvae_dataset}


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

    def visualization(self, recon_state, state, occupancy, occu_map_path_batch, depth, W_world, H_world, idx):
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
            heading = np.rad2deg(input_state[2] * (2 * math.pi))
            vis.draw_location(world_pos, heading, 'r', markersize=7)

            # map the reconstructed results and draw it with GREEN color
            output_state = recon_state[idy].detach().cpu().numpy()
            world_pos = np.array([output_state[0] * W_world[idy], output_state[1] * H_world[idy]])
            heading = np.rad2deg(output_state[2] * (2 * math.pi))
            vis.draw_location(world_pos, heading, 'g', markersize=7)

            # convert the matplotlib object to PIL.Image object and save it to the desired dir
            img = fig2img(fig)
            img.save("{}/{:03d}.png".format(vis_dir, idx * self.cfg.data.batch_size + idy + 1))

            if self.cfg.heatmap.gen:
                if self.cfg.heatmap.num_data <= 256:
                    occupancy_repeat = occupancy[idy].unsqueeze(0).repeat(self.cfg.heatmap.num_data, 1, 1, 1)
                    depth_repeat = depth[idy].unsqueeze(0).repeat(self.cfg.heatmap.num_data, 1)
                    # forward
                    with torch.no_grad():
                        recon_state_repeat = self.model.inference(occupancy_repeat, depth_repeat)
                else:
                    num_iters = self.cfg.heatmap.num_data // 256 + ((self.cfg.heatmap.num_data % 256) > 0)
                    recon_state_repeat = []
                    for j in range(num_iters):
                        occupancy_repeat = occupancy[idy].unsqueeze(0).repeat(256, 1, 1, 1)
                        depth_repeat = depth[idy].unsqueeze(0).repeat(256, 1)
                        with torch.no_grad():
                            recon_state_tmp = self.model.inference(occupancy_repeat, depth_repeat)
                            recon_state_repeat.append(recon_state_tmp.detach())
                    recon_state_repeat = torch.cat(recon_state_repeat)

                # map the reconstructed results and draw it with GREEN color
                output_state = recon_state_repeat.detach().cpu().numpy()
                world_pos = np.array([output_state[:, 0] * W_world[idy], output_state[:, 1] * H_world[idy]])
                vis.draw_heatmap(world_pos)
                img = fig2img(fig)
                img.save("{}/heatmap_{:03d}.png".format(vis_dir, idx * self.cfg.data.batch_size + idy + 1))


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
                self.visualization(recon_state, state, occupancy, occu_map_path_batch, depth, W_world, H_world, idx)

        print("finish!")
        print("state reconstruction BCE loss: {:.3f}±{:.3f}".format(np.mean(recon_losses), np.std(recon_losses)))
        json.dump(
            {
                "recon_BCE_Loss": np.array(recon_losses).tolist(),
            },
            self.logger,
        )
        self.logger.close()


class Tester_cls(Tester):
    def __init__(self, configs):
        super(Tester_cls, self).__init__(configs)

    def loss(self, pred, gt):
        gt = torch.flatten(gt)
        pred = pred.view(-1, self.cfg.data.num_bin)
        return nn.functional.cross_entropy(pred, gt, reduction='mean')

    def get_correct(self, pred, gt):
        pred = pred.argmax(1)
        correct = []
        for idx in range(len(pred)):
            if pred[idx, 0] == gt[idx, 0] and pred[idx, 1] == gt[idx, 1] and pred[idx, 2] == gt[idx, 2]:
                correct.append(True)
            else:
                correct.append(False)
        return correct

    def run(self):
        losses, total_correct = [], []
        for idx, (occupancy, depth, state, occu_map_path_batch, W_world, H_world) in enumerate(self.dataloader):

            if self.cfg.framework.num_gpu > 0:
                occupancy, depth, state, W_world, H_world, ones = (
                    occupancy.to(device=0),
                    depth.to(device=0),
                    state.to(device=0),
                    W_world.to(device=0),
                    H_world.to(device=0),
                    torch.ones(1, dtype=torch.float64).to(device=0)
                )

            # convert state to classes
            world_pos_x_cls = ((state[:, 0] * W_world) // (W_world / self.cfg.data.num_bin)).unsqueeze(1)
            world_pos_y_cls = ((state[:, 1] * H_world) // (H_world / self.cfg.data.num_bin)).unsqueeze(1)
            heading_cls = ((state[:, 2]) // (ones / self.cfg.data.num_bin)).unsqueeze(1)
            gt = torch.cat((world_pos_x_cls, world_pos_y_cls, heading_cls), 1).long()

            # forward
            with torch.no_grad():
                pred = self.model(occupancy, depth)

            # loss
            l = self.loss(pred, gt)
            correct = self.get_correct(pred, gt)
            total_correct += correct

            ## save occupancy, state, reconstructed state for visualization later

            losses.append(l.detach().cpu().numpy())

            if self.cfg.vis:
                # convert predicted classes back to the world coordinate
                pred, W_world, H_world = pred.detach().cpu(), W_world.cpu(), H_world.cpu()
                world_pos_x_pred = (pred[:, 0, :].argmax() * (W_world / self.cfg.data.num_bin)).unsqueeze(1)
                world_pos_x_pred += ((W_world / self.cfg.data.num_bin) / 2)
                world_pos_x_pred /= W_world
                world_pos_y_pred = (pred[:, 1, :].argmax() * (H_world / self.cfg.data.num_bin)).unsqueeze(1)
                world_pos_y_pred += ((H_world / self.cfg.data.num_bin) / 2)
                world_pos_y_pred /= H_world
                world_heading_pred = (pred[:, 2, :].argmax() *
                                      (torch.ones(1, dtype=torch.float64) / self.cfg.data.num_bin)).unsqueeze(1)
                world_heading_pred += ((torch.ones(1, dtype=torch.float64) / self.cfg.data.num_bin) / 2)
                recon_state = torch.cat((world_pos_x_pred, world_pos_y_pred, world_heading_pred), 1)

                self.visualization(recon_state, state, occupancy, occu_map_path_batch, depth, W_world, H_world, idx)

        print("finish!")
        print("state cls loss: {:.3f}±{:.3f}".format(np.mean(losses), np.std(losses)))
        print("cls acc: {:.3f}".format(np.mean(total_correct)))
        json.dump(
            {
                "cls_Loss": np.array(losses).tolist(),
                "cls_correct": total_correct,
            },
            self.logger,
        )
        self.logger.close()


class Tester_reg(Tester_cvae):
    def __init__(self, configs):
        super(Tester_reg, self).__init__(configs)

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
                recon_state = self.model(occupancy, depth)

            # loss
            l = self.BCE_loss(recon_state, state)

            ## save occupancy, state, reconstructed state for visualization later

            recon_losses.append(l.detach().cpu().numpy())

            if self.cfg.vis:
                self.visualization(recon_state, state, occupancy, occu_map_path_batch, depth, W_world, H_world, idx)

        print("finish!")
        print("state reconstruction BCE loss: {:.3f}±{:.3f}".format(np.mean(recon_losses), np.std(recon_losses)))
        json.dump(
            {
                "recon_BCE_Loss": np.array(recon_losses).tolist(),
            },
            self.logger,
        )
        self.logger.close()
