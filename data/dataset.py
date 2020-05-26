import torch
import numpy as np
from torch.utils.data import Dataset
import utils.map_utils
import glob, os.path as osp
import random
class cvae_dataset(Dataset):
    def __init__(self, cfg):

        # store useful things ...
        self.cfg = cfg
        self.crop_size = cfg.data.crop_size
        self.downsample_factor = cfg.data.downsample_factor
        self.laser_max_range = cfg.data.laser_max_range
        self.n_ray = cfg.data.n_ray
        self.fov = cfg.data.fov

        self.occu_map_paths= sorted(glob.glob(osp.join(cfg.data.occu_map_dir,'*')))
        self.length = len(self.occu_map_paths)


    def __len__(self):
        return self.length

    def __getitem__(self, index):

        occu_map_path = self.occu_map_paths[index]

        ## load map and random crop
        m = Map(osp.join(occu_map_path, 'floorplan.yaml'), 
                laser_max_range=self.laser_max_range, 
                 downsample_factor=self.downsample_factor,
                 crop_size = self.crop_size)


        ## uniformly random sample state in freespace
        while m.get_occupancy_value(pos_x, pos_y) != 0:
            pos_x = random.randint(0, self.crop_size)
            pos_y = random.randint(0, self.crop_size)

        pos = np.array([pos_x, pos_y])*self.resolution ## pos coords in meters
        heading = np.deg2rad(90) ## fix heading
        fov = np.deg2rad(self.fov)
        depth = m.get_1d_depth(pos, heading, fov, self.n_ray, resolution=0.01)
        depth_xy = depth_to_xy(depth, pos, heading, fov)

        state = np.array([pos[0], pos[1], heading])

        return torch.Tensor(m.occupancy_grid), torch.Tensor(depth_xy), torch.Tensor(state)