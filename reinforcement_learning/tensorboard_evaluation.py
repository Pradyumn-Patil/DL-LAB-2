from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

class Evaluation:
    def __init__(self, store_dir, name, stats=[]):
        """
        Creates a Tensorboard writer instance to log training progress
        
        Args:
            store_dir (str): Path to store tensorboard files
            name (str): Name of this training run
            stats (list): List of statistics names to track
        """
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
            
        self.stats = stats
        self.writer = SummaryWriter(os.path.join(store_dir, name))

    def write_episode_data(self, episode, eval_dict):
        """
        Write episode statistics to tensorboard
        
        Args:
            episode (int): Episode number
            eval_dict (dict): Dictionary containing statistics
        """
        for key in eval_dict:
            if key in self.stats:
                self.writer.add_scalar(key, eval_dict[key], episode)

    def close_session(self):
        """
        Close the tensorboard writer
        """
        self.writer.close()
