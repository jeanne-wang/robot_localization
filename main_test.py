from scripts.experiments_helper import tester_protocol, get_configs


def main(cfg):

    worker = tester_protocol[cfg.exp_prefix](cfg)
    worker.run()


if __name__ == "__main__":
    config, _ = get_configs()
    main(config)