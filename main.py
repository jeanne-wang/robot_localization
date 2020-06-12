from scripts.experiments_helper import trainer_protocol, tester_protocol, get_configs


def main(cfg):
    if "train" in cfg.mode:
        worker = trainer_protocol[cfg.exp_prefix](cfg)
    elif "test" in cfg.mode:
        worker = tester_protocol[cfg.exp_prefix](cfg)
    else:
        raise NotImplementedError
    worker.run()

if __name__ == "__main__":
    config, _ = get_configs()
    main(config)