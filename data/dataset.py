import torch
import numpy as np
from torch.utils.data import Dataset
from utils.map_utils import Map, depth_to_xy, Visualizer
from utils.utils import fig2data, AStarPlanner, MapEnvironment
import glob, os.path as osp
import random
import math, time
import matplotlib.pyplot as plt

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
        if self.crop_size == 0:
            m = Map(osp.join(occu_map_path, 'floorplan.yaml'), 
                    laser_max_range=self.laser_max_range, 
                    downsample_factor=self.downsample_factor)
        else:
            m = Map(osp.join(occu_map_path, 'floorplan.yaml'), 
                    laser_max_range=self.laser_max_range, 
                    downsample_factor=self.downsample_factor,
                    crop_size=self.crop_size)

        occupancy = m.get_occupancy_grid()
        W_world = occupancy.shape[1]*m.resolution
        H_world = occupancy.shape[0]*m.resolution

        ## uniformly random sample state in freespace
        while True:
            world_pos_x = np.random.uniform(0, W_world, 1)[0]
            world_pos_y = np.random.uniform(0, H_world, 1)[0]
            grid_pos_x, grid_pos_y = m.grid_coord(world_pos_x, world_pos_y)
            grid_pos_x = np.clip(grid_pos_x, 0, occupancy.shape[1]-1)
            grid_pos_y = np.clip(grid_pos_y, 0, occupancy.shape[0]-1)

            if occupancy[grid_pos_y, grid_pos_x]== 0:
                break
        
        world_pos = np.array([world_pos_x, world_pos_y])
        ## uniformly random sample heading
        heading = np.random.uniform(0, 360, 1)[0]
        heading = np.deg2rad(heading)
        depth = m.get_1d_depth(world_pos, heading, self.fov, self.n_ray)
        depth_xy = depth_to_xy(depth, world_pos, heading, self.fov)

        ## clip depth
        np.clip(depth_xy[:, 0], 0, W_world, out=depth_xy[:, 0])
        np.clip(depth_xy[:, 1], 0, H_world, out=depth_xy[:, 1])

        ## normalize state position and depth info to [0,1]
        if self.cfg.data.world_coord_laser:
            depth_xy[:, 0] = (depth_xy[:, 0]) / W_world
            depth_xy[:, 1] = (depth_xy[:, 1]) / H_world
        else:
            depth_xy[:,0] = (depth_xy[:,0] - world_pos_x)/W_world
            depth_xy[:,1] = (depth_xy[:,1] - world_pos_y)/H_world
        world_pos_x = world_pos_x/W_world
        world_pos_y = world_pos_y/H_world
        heading = heading/(2*math.pi)

    
        state = torch.Tensor(np.array([world_pos_x, world_pos_y, heading]))
        occupancy = torch.Tensor(occupancy).unsqueeze(0) ## change shape to (1, W, H)
        depth_xy = torch.Tensor(depth_xy).view(-1)

        return occupancy, depth_xy, state, occu_map_path, W_world, H_world


class seq_cvae_dataset(Dataset):
    def __init__(self, cfg):

        # store useful things ...
        self.cfg = cfg
        self.crop_size = cfg.data.crop_size
        self.downsample_factor = cfg.data.downsample_factor
        self.laser_max_range = cfg.data.laser_max_range
        self.n_ray = cfg.data.n_ray
        self.fov = np.deg2rad(cfg.data.fov)

        self.occu_map_paths = sorted(glob.glob(osp.join(cfg.data.occu_map_dir, '*')))
        self.length = len(self.occu_map_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        good_to_go = False
        while not good_to_go:
            s = time.time()
            timeout = False
            if self.cfg.mode == "train":
                occu_map_path = np.random.choice(self.occu_map_paths)
            else:
                occu_map_path = self.occu_map_paths[index]
            #occu_map_path = self.occu_map_paths[index]

            ## load map and random crop
            if self.crop_size == 0:
                m = Map(osp.join(occu_map_path, 'floorplan.yaml'),
                        laser_max_range=self.laser_max_range,
                        downsample_factor=self.downsample_factor)
            else:
                m = Map(osp.join(occu_map_path, 'floorplan.yaml'),
                        laser_max_range=self.laser_max_range,
                        downsample_factor=self.downsample_factor,
                        crop_size=self.crop_size)

            occupancy = m.get_occupancy_grid()
            W_world = occupancy.shape[1] * m.resolution
            H_world = occupancy.shape[0] * m.resolution

            ## uniformly random sample state in freespace
            while True:
                if time.time() - s > 10:
                    timeout = True
                    break
                while True:
                    world_pos_x = np.random.uniform(0, W_world, 1)[0]
                    world_pos_y = np.random.uniform(0, H_world, 1)[0]
                    grid_pos_x, grid_pos_y = m.grid_coord(world_pos_x, world_pos_y)
                    grid_pos_x = np.clip(grid_pos_x, 0, occupancy.shape[1] - 1)
                    grid_pos_y = np.clip(grid_pos_y, 0, occupancy.shape[0] - 1)

                    if occupancy[grid_pos_y, grid_pos_x] == 0:
                        break
                    if time.time() - s > 10:
                        timeout = True
                        break
                if timeout:
                    break

                while True:
                    goal_world_pos_x = np.random.uniform(0, W_world, 1)[0]
                    goal_world_pos_y = np.random.uniform(0, H_world, 1)[0]
                    goal_grid_pos_x, goal_grid_pos_y = m.grid_coord(goal_world_pos_x, goal_world_pos_y)
                    goal_grid_pos_x = np.clip(goal_grid_pos_x, 0, occupancy.shape[1] - 1)
                    goal_grid_pos_y = np.clip(goal_grid_pos_y, 0, occupancy.shape[0] - 1)

                    if occupancy[goal_grid_pos_y, goal_grid_pos_x] == 0:
                        break
                    if time.time() - s > 10:
                        timeout = True
                        break
                if timeout:
                    break

                start_config = np.array([[grid_pos_x], [grid_pos_y]])
                goal_config = np.array([[goal_grid_pos_x], [goal_grid_pos_y]])
                env = MapEnvironment(occupancy, start_config, goal_config, self.cfg.data.epsilon)
                planner = AStarPlanner(env, 10)
                plan = planner.Plan(start_config, goal_config)
                if plan.shape[1] > self.cfg.data.horizon:
                    good_to_go = True
                    break
            if not timeout:
                break


        depth_xy_seq, state_seq = [], []
        heading = np.random.uniform(0, 360, 1)[0]
        heading = np.deg2rad(heading)
        heading_normalized = heading / (2 * math.pi)
        for i in range(self.cfg.data.horizon):
            world_pos_x, world_pos_y = m.world_coord(plan[0, i], plan[1, i])

            world_pos = np.array([world_pos_x, world_pos_y])
            ## uniformly random sample heading
            depth = m.get_1d_depth(world_pos, heading, self.fov, self.n_ray)
            depth_xy = depth_to_xy(depth, world_pos, heading, self.fov)

            ## clip depth
            np.clip(depth_xy[:, 0], 0, W_world, out=depth_xy[:, 0])
            np.clip(depth_xy[:, 1], 0, H_world, out=depth_xy[:, 1])

            ## normalize state position and depth info to [0,1]
            if self.cfg.data.world_coord_laser:
                depth_xy[:, 0] = (depth_xy[:, 0]) / W_world
                depth_xy[:, 1] = (depth_xy[:, 1]) / H_world
            else:
                depth_xy[:, 0] = (depth_xy[:, 0] - world_pos_x) / W_world
                depth_xy[:, 1] = (depth_xy[:, 1] - world_pos_y) / H_world
            world_pos_x = world_pos_x / W_world
            world_pos_y = world_pos_y / H_world

            state = torch.Tensor(np.array([world_pos_x, world_pos_y, heading_normalized]))
            depth_xy = torch.Tensor(depth_xy).view(-1)

            depth_xy_seq.append(depth_xy.unsqueeze(0))
            state_seq.append(state.unsqueeze(0))
        depth_xy_seq = torch.cat(depth_xy_seq)
        state_seq = torch.cat(state_seq)

        occupancy = torch.Tensor(occupancy).unsqueeze(0)  ## change shape to (1, W, H)

        return occupancy, depth_xy_seq, state_seq, occu_map_path, W_world, H_world


class cnn_dataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.a = cfg.model
        
        self.crop_size = cfg.data.crop_size
        self.downsample_factor = cfg.data.downsample_factor
        self.laser_max_range = cfg.data.laser_max_range
        self.n_ray = cfg.data.n_ray
        self.fov = cfg.data.fov

        self.num_bin = cfg.data.num_bin

        self.occu_map_paths= sorted(glob.glob(osp.join(cfg.data.occu_map_dir,'*')))
        self.length = len(self.occu_map_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        occu_map_path = self.occu_map_paths[index]

        ## load map and random crop
        if self.crop_size == 0:
            m = Map(osp.join(occu_map_path, 'floorplan.yaml'),
                    laser_max_range=self.laser_max_range,
                    downsample_factor=self.downsample_factor)
        else:
            m = Map(osp.join(occu_map_path, 'floorplan.yaml'),
                    laser_max_range=self.laser_max_range,
                    downsample_factor=self.downsample_factor,
                    crop_size=self.crop_size)

        occupancy = m.get_occupancy_grid()
        W_world = occupancy.shape[1] * m.resolution
        H_world = occupancy.shape[0] * m.resolution

        ## uniformly random sample state in freespace
        while True:
            world_pos_x = np.random.uniform(0, W_world, 1)[0]
            world_pos_y = np.random.uniform(0, H_world, 1)[0]
            grid_pos_x, grid_pos_y = m.grid_coord(world_pos_x, world_pos_y)
            grid_pos_x = np.clip(grid_pos_x, 0, occupancy.shape[1] - 1)
            grid_pos_y = np.clip(grid_pos_y, 0, occupancy.shape[0] - 1)

            if occupancy[grid_pos_y, grid_pos_x] == 0:
                break

        world_pos = np.array([world_pos_x, world_pos_y])
        ## uniformly random sample heading
        heading = np.random.uniform(0, 360, 1)[0]
        heading = np.deg2rad(heading)
        depth = m.get_1d_depth(world_pos, heading, self.fov, self.n_ray)
        depth_xy = depth_to_xy(depth, world_pos, heading, self.fov)

        ## clip depth
        np.clip(depth_xy[:, 0], 0, W_world, out=depth_xy[:, 0])
        np.clip(depth_xy[:, 1], 0, H_world, out=depth_xy[:, 1])

        ## normalize state position and depth info to [0,1]
        if self.cfg.data.world_coord_laser:
            depth_xy[:, 0] = (depth_xy[:, 0]) / W_world
            depth_xy[:, 1] = (depth_xy[:, 1]) / H_world
        else:
            depth_xy[:,0] = (depth_xy[:,0] - world_pos_x)/W_world
            depth_xy[:,1] = (depth_xy[:,1] - world_pos_y)/H_world
        world_pos_x_cls = (world_pos_x) // (W_world / self.num_bin)
        world_pos_y_cls = (world_pos_y) // (H_world / self.num_bin)
        heading = heading / (2 * math.pi)
        heading_cls = (heading) // (1 / self.num_bin)
        cls = np.array([world_pos_x_cls, world_pos_y_cls, heading_cls])

        cls = torch.LongTensor(cls)
        occupancy = torch.Tensor(occupancy).unsqueeze(0) ## change shape to (1, W, H)
        depth_xy = torch.Tensor(depth_xy).view(-1)

        return occupancy, depth_xy, cls, occu_map_path, W_world, H_world
