import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Evaluation:

    def __init__(self, store_dir, name, stats=None):
        """
        stats: list of strings containing statistics that should be tracked
        """
        self.store_dir = store_dir
        self.stats = stats
        self.writer = SummaryWriter(os.path.join(store_dir, name))

    def write_episode_data(self, episode, eval_dict):
        """
        writes episode statistics in eval_dict to tensorboard
        """

        # Check for missing keys in eval_dict
        if not all(k in self.stats for k in eval_dict):
            raise AssertionError(f"Missing key from {self.stats} in {eval_dict.keys()}")

        # Write all statistics to tensorboard
        for key in eval_dict:
            self.writer.add_scalar(key, eval_dict[key], episode)

    def close_session(self):
        self.writer.close()
