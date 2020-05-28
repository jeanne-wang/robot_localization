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
        self.fov = np.deg2rad(cfg.data.fov)

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
                 crop_size=self.crop_size)

        occupancy = m.get_occupancy_grid()
        W_world = occupancy.shape[1]*m.resolution
        H_world = occupancy.shape[0]*m.resolution

        ## uniformly random sample state in freespace
        while occupancy[grid_pos_y, grid_pos_x]!= 0:
            world_pos_x = np.random.uniform(0, W_world, 1)[0]
            world_pos_y = np.random.uniform(0, H_world, 1)[0]
            grid_pos_x, grid_pos_y = m.grid_coord(world_pos_x, world_pos_y)
            grid_pos_x = np.clip(grid_pos_x, 0, occupancy.shape[1]-1)
            grid_pos_y = np.clip(grid_pos_y, 0, occupancy.shape[0]-1)
        
        world_pos = np.array((world_pos_x, world_pos_y))
        ## uniformly random sample heading
        heading = np.random.uniform(0, 360, 1)[0]
        heading = np.deg2rad(heading)
        depth = m.get_1d_depth(world_pos, heading, self.fov, self.n_ray)
        depth_xy = depth_to_xy(depth, world_pos, heading, self.fov)

        ## normalize state position and depth info to [-1,1]
        world_pos_x = 2.0*world_pos_x/ max(W_world-1,1)-1.0
        world_pos_y = 2.0*world_pos_y/ max(H_world-1,1)-1.0
        depth_xy[:,0] = 2.0*depth_xy[:,0]/ max(W_world-1,1)-1.0
        depth_xy[:,1] = 2.0*depth_xy[:,1]/ max(H_world-1,1)-1.0
        heading = 2*heading/(2*math.pi-1)-1.0

        print(state)
        print(depth_xy)
        state = torch.Tensor(np.array((world_pos_x, world_pos_y, heading)))
        occupancy = torch.Tensor(occupancy).unsqueeze(0)
        depth_xy = torch.Tensor(depth_xy).view(-1,)
        
        print(depth_xy.shape)
        print(state.shape)

        return occupancy, depth_xy, state

    
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
                # heading = np.random.uniform(0,np.pi/4) + np.pi/4*class_pos[2]
                heading = class_pos[2] * 90
                
                grid_pos_x, grid_pos_y = m.grid_coord(world_pos[0] + class_pos[0] *class_size  , world_pos[1]+ class_pos[1] *class_size)
                label = (1 + class_pos[0]) * 100 + (1+class_pos[1]) * 10 + (1 + class_pos[2])
                occupancy = occupancy_grid[grid_pos_x, grid_pos_y]
                fov = np.deg2rad(self.fov)
                depth = m.get_1d_depth(world_pos, heading, fov, self.n_ray)
                depth_xy = depth_to_xy(depth, world_pos, heading, fov)
                
                #save the class info into state
                state = np.array([class_pos[0]/4, class_pos[1]/4, heading,label])
                state_arr = np.append(state_arr,state)
                depth_xy_arr = np.append(depth_xy_arr,depth_xy)
            # keep finding next state
            occupancy = 1
            

        state = state_arr.reshape(-1,4)
        depth_xy = depth_xy_arr.reshape(-1,1)
        print(torch.Tensor(state))
        return torch.Tensor(occupancy_grid).unsqueeze(0), torch.Tensor(depth_xy).view(self.n_ray, -1), torch.Tensor(state)
    
