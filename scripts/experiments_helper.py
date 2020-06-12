import argparse
from utils.config import Config
from utils.config import Replaced_Config
from scripts import trainer, tester
from multiprocessing import Process
import os, torch
import numpy as np

trainer_protocol = {
    "cvae": trainer.Trainer_cvae,
    "cls": trainer.Trainer_cls,
    "reg": trainer.Trainer_reg,
    "seq_cvae": trainer.Trainer_cvae,
}

tester_protocol = {
    "cvae": tester.Tester_cvae,
    "cls": tester.Tester_cls,
    "reg": tester.Tester_reg,
    "seq_cvae": tester.Tester_cvae,
}

def get_configs(replace=True):
    parser = argparse.ArgumentParser(description="We are the best!!!")
    parser.add_argument("--config", type=str, default="configs/cvae_train.yaml")
    parser.add_argument("--n", type=int, default=2)
    args = parser.parse_args()
    config = Config(args.config, replace)
    return config, args.n


def get_all_possible_experiments_options(options, queue):
    length = []
    for v in options.values():
        length.append(len(v))

    new_options = {}
    for i, (k, v) in enumerate(options.items()):
        v = np.array(v)

        new_shape = np.ones(len(options.keys()), dtype=int)
        new_shape[i] = length[i]
        v = v.reshape(new_shape)

        axis = list(range(len(options.keys())))
        axis.remove(i)
        for a in axis:
            v = np.repeat(v, length[a], axis=a)

        new_options[k] = v.flatten()

    keys = new_options.keys()
    for n in range(np.prod(length)):
        msg = {}
        for k in keys:
            msg[k] = new_options[k][n]
        queue.put(msg)

    return queue


class Worker(Process):
    def __init__(self, process_id, config, queue):
        super(Worker, self).__init__()
        self.id = process_id
        self.gpu_id = self.id % torch.cuda.device_count()
        self.config = config
        self.queue = queue

    def replace_recursively(self, yaml, key, value):
        k = key.split(".")[0]
        if isinstance(yaml[k], dict):
            new_key = ".".join(key.split(".")[1:])
            yaml[k] = self.replace_recursively(yaml[k], new_key, value)
        else:
            if isinstance(value, float):
                yaml[k] = float(value)
            elif isinstance(value, int):
                yaml[k] = int(value)
            elif isinstance(value, np.int64):
                yaml[k] = int(value)
            else:
                yaml[k] = str(value)
        return yaml

    def replace_options(self):
        yaml = self.config.yaml()
        options = self.queue.get()
        yaml["base_dir"] = "results/{{dir_prefix}}/"
        for k, v in options.items():
            yaml["base_dir"] = "{}_{{{{{}}}}}".format(yaml["base_dir"], k)
        for k, v in options.items():
            yaml = self.replace_recursively(yaml, k, v)
        return Replaced_Config(yaml)

    def save_yaml(self, cfg):
        save_dir = "{}/mp_state.yaml".format(cfg.base_dir)
        if not os.path.isdir(cfg.base_dir):
            os.makedirs(cfg.base_dir)
        cfg.save(save_dir)

    def run(self):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        os.environ.update(env)
        while not self.queue.empty():
            new_config = self.replace_options()
            self.save_yaml(new_config)
            worker = trainer_protocol[new_config.exp_prefix](new_config)
            worker.run()
            worker = tester_protocol[new_config.exp_prefix](new_config)
            worker.run()
        print("process id {} finish!".format(self.id))
