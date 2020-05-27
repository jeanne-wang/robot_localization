import torch
import numpy as np
from torch.utils.data import Dataset
from utils.map_utils import Map, depth_to_xy
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

        occupancy= m.get_occupancy_grid()
        ## uniformly random sample state in freespace
        while occupancy[grid_pos_x, grid_pos_y]!= 0:
            world_pos = np.random.uniform(0, self.crop_size*m.resolution, 2)
            grid_pos_x, grid_pos_y = m.grid_coord(world_pos[0], world_pos[1])
        

        heading = np.deg2rad(90) ## fix heading
        fov = np.deg2rad(self.fov)
        depth = m.get_1d_depth(world_pos, heading, fov, self.n_ray)
        depth_xy = depth_to_xy(depth, world_pos, heading, fov)

        state = np.array([world_pos[0], world_pos[1], heading])
        
        return torch.Tensor(occupancy).unsqueeze(0), torch.Tensor(depth_xy).view(self.n_ray, -1), torch.Tensor(state)

class cnn_dataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.a = cfg.model
        
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

        occupancy_grid = m.get_occupancy_grid()
        ## uniformly random sample state in freespace
        occupancy = 1 #0 means no obstacle
        state_arr = np.array([])
        depth_xy_arr = np.array([])
        for i in range(100):
            # sample state in freespace
            while occupancy != 0:
                class_size = self.crop_size*m.resolution/4
                class_pos = np.random.randint(0,4,3)
                world_pos = np.random.uniform(0, class_size, 2)
                heading = np.random.uniform(0,np.pi/4) + np.pi/4*class_pos[2]
                grid_pos_x, grid_pos_y = m.grid_coord(world_pos[0] + class_pos[0] *class_size  , world_pos[1]+ class_pos[1] *class_size)
                label = (1 + class_pos[0]) * 100 + (1+class_pos[1]) * 10 + (1 + class_pos[2])
                occupancy = occupancy_grid[grid_pos_x, grid_pos_y]
                fov = np.deg2rad(self.fov)
                depth = m.get_1d_depth(world_pos, heading, fov, self.n_ray)
                depth_xy = depth_to_xy(depth, world_pos, heading, fov)
                state = np.array([world_pos[0], world_pos[1], heading,label])
                state_arr = np.append(state_arr,state)
                depth_xy_arr = np.append(depth_xy_arr,depth_xy)
            # keep finding next state
            occupancy = 1
            

        state = state_arr.reshape(-1,4)
        depth_xy = depth_xy_arr.reshape(-1,1)
        print(torch.Tensor(state))
        return torch.Tensor(occupancy_grid).unsqueeze(0), torch.Tensor(depth_xy).view(self.n_ray, -1), torch.Tensor(state)
    
