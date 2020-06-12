import numpy, time
from PIL import Image

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = numpy.fromstring(fig.canvas.tostring_argb(), dtype=numpy.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = numpy.roll(buf, 3, axis=2)
    return buf

def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

def data2img(buf):
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())

import numpy as np


class AStarPlanner(object):
    def __init__(self, planning_env, epsilon):
        self.env = planning_env
        self.nodes = {}
        self.epsilon = epsilon
        self.visited = np.zeros(self.env.map.shape)

    def Plan(self, start_config, goal_config):
        # TODO: YOUR IMPLEMENTATION HERE

        q = [start_config]
        c = [0]
        a_c = [0]
        while len(q) > 0:
            current = q.pop(0)
            _ = c.pop(0)
            current_action_cost = a_c.pop(0)
            last_x, last_y = current[:, -1]
            if not self.visited[int(last_y), int(last_x)]:
                self.visited[int(last_y), int(last_x)] = 1
                if self.env.goal_criterion(np.array([[last_x], [last_y]]), goal_config):
                    break
                next_states = []
                next_cost = []
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        next_states.append(np.array([[last_x + i], [last_y + j]]))
                        next_cost.append(self.env.compute_distance(np.array([[last_x + i], [last_y + j]]),
                                                                   np.array([[last_x], [last_y]])))
                for i, state in enumerate(next_states):
                    if self.env.state_validity_checker(state):
                        q.append(np.hstack([current, state]))
                        cost = current_action_cost + next_cost[i]
                        a_c.append(cost)
                        cost += self.epsilon * self.env.h(np.hstack([state, goal_config]))
                        c.append(cost)
                idx = sorted(range(len(c)), key=lambda k: c[k])
                q = [q[idy] for idy in idx]
                c = [c[idy] for idy in idx]
                a_c = [a_c[idy] for idy in idx]
        plan = current.tolist()

        return np.array(plan)


class MapEnvironment(object):

    def __init__(self, map, start, goal, epsilon=0.01):

        # Obtain the boundary limits
        self.map = map
        self.xlimit = [0, np.shape(self.map)[1] - 1]
        self.ylimit = [0, np.shape(self.map)[0] - 1]

        self.goal = goal
        self.epsilon = epsilon

        # Check if start and goal are within limits and collision free
        if not self.state_validity_checker(start) or not self.state_validity_checker(goal):
            raise ValueError('Start and Goal state must be within the map limits');
            exit(0)

    def sample(self):
        # Sample random clear point from map
        clear = np.argwhere(self.map == 0)
        idx = np.random.choice(len(clear))
        return clear[idx, :].reshape((2, 1))

    def goal_criterion(self, config, goal_config):
        """ Return True if config is close enough to goal

            @param config: a [2 x 1] numpy array of a state
            @param goal_config: a [2 x 1] numpy array of goal state
        """
        return self.compute_distance(config, goal_config) < self.epsilon

    def compute_distance(self, start_config, end_config):
        """ A function which computes the distance between
            two configurations.

            @param start_config: a [2 x 1] numpy array of current state
            @param end_config: a [2 x 1] numpy array of goal state
        """
        # TODO: YOUR IMPLEMENTATION HERE
        dis = ((start_config - end_config) ** 2).sum() ** 0.5
        return dis

    def state_validity_checker(self, config):
        """ Return True if all states are valid

            @param config: a [2 x n] numpy array of states
        """
        # TODO: YOUR IMPLEMENTATION HERE
        for i in range(config.shape[1]):
            if config[0, i] < self.xlimit[0] or config[0, i] > self.xlimit[1]:
                return False
            if config[1, i] < self.ylimit[0] or config[1, i] > self.ylimit[1]:
                return False
            if self.map[int(config[1, i]), int(config[0, i])] == 1.0:
                return False
        return True

    def edge_validity_checker(self, config1, config2):
        """ Return True if edge is valid

            @param config1: a [2 x 1] numpy array of state
            @param config2: a [2 x 1] numpy array of state
        """
        assert (config1.shape == (2, 1))
        assert (config2.shape == (2, 1))
        n = max(self.xlimit[1], self.ylimit[1])
        x_vals = np.linspace(config1[0], config2[0], n).reshape(1, n)
        y_vals = np.linspace(config1[1], config2[1], n).reshape(1, n)
        configs = np.vstack((x_vals, y_vals))
        return self.state_validity_checker(configs)

    def h(self, config):
        """ Heuristic function for A*

            @param config: a [2 x 1] numpy array of state
        """
        # TODO: YOUR IMPLEMENTATION HERE
        return self.compute_distance(np.array([[config[0, 0]], [config[1, 0]]]),
                                     np.array([[config[0, 1]], [config[1, 1]]]))
