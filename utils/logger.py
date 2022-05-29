"""Implementation of a Logger class. Adapted from RLLib"""
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import wandb


def safe_make_dir(dir_name):
    """Create a new directory safely."""
    try:
        os.makedirs(dir_name)
    except OSError:
        now = datetime.now()
        dir_name = safe_make_dir(dir_name + f"-{now.microsecond}")
    return dir_name


class Logger(object):
    """Class that implements a logger of statistics.

    Parameters
    ----------
    name: str
        Name of logger. This create a folder at runs/`name'.
    comment: str, optional.
        This is useful to separate equivalent runs.
        The folder is runs/`name'/`comment_date'.
    tensorboard: bool, optional.
        Flag that indicates whether or not to save the results in the tensorboard.
    """

    def __init__(self, name, comment="", use_wandb=True):
        self.statistics = list()
        self.current = dict()
        self.all = defaultdict(list)
        self.use_wandb = use_wandb

        now = datetime.now()
        current_time = now.strftime("%b%d_%H-%M-%S")
        comment = comment + "_" + current_time if len(comment) else current_time
        log_dir = f"runs/{name}/{comment}"
        if use_wandb:
            wandb.init(
                project="meta-mbrl",
                name=name,
                notes=comment
            )
        self.log_dir = safe_make_dir(log_dir)
        self.episode = 0
        self.keys = set()

    def __len__(self):
        """Return the number of episodes."""
        return len(self.statistics)

    def __iter__(self):
        """Iterate over the episode statistics."""
        return self.statistics

    def __getitem__(self, index):
        """Return a specific episode."""
        return self.statistics[index]

    def __str__(self):
        """Return parameter string of logger."""
        str_ = ""
        for key in sorted(self.keys):
            values = self.get(key)
            str_ += " ".join(key.split("_")).title().ljust(17)
            str_ += f"Last: {values[-1]:.2g}".ljust(15)
            str_ += f"Avg: {np.mean(values):.2g}".ljust(15)
            str_ += f"MAvg: {np.mean(values[-10:]):.2g}".ljust(15)
            str_ += f"Range: ({np.min(values):.2g},{np.max(values):.2g})\n"

        return str_

    def get(self, key):
        """Return the statistics of a specific key.

        It collects all end-of-episode data stored in statistic and returns a list with
        such values.
        """
        return [statistic[key] for statistic in self.statistics if key in statistic]

    def update(self, **kwargs):
        """Update the statistics for the current episode.

        Parameters
        ----------
        kwargs: dict
            Any kwargs passed to update is converted to numpy and averaged
            over the course of an episode.
        """
        for key, value in kwargs.items():
            self.keys.add(key)
            if isinstance(value, torch.Tensor):
                value = value.detach().numpy()
            value = np.nan_to_num(value)
            if isinstance(value, np.ndarray):
                value = float(np.mean(value))
            if isinstance(value, np.float32):
                value = float(value)
            if isinstance(value, np.int64):
                value = int(value)

            if key not in self.current:
                self.current[key] = (1, value)
            else:
                count, old_value = self.current[key]
                new_count = count + 1
                new_value = old_value + (value - old_value) * (1 / new_count)
                self.current[key] = (new_count, new_value)

            self.all[key].append(value)

            # if self.use_wandb is not None:
            #     wandb.log(
            #         {f"episode_{self.episode}/{key}": self.current[key][1]},
            #         step=self.current[key][0],
            #     )

    def end_episode(self, **kwargs):
        """Finalize collected data and add final fixed values.

        Parameters
        ----------
        kwargs : dict
            Any kwargs passed to end_episode overwrites tracked data if present.
            This can be used to store fixed values that are tracked per episode
            and do not need to be averaged.
        """
        data = {key: value[1] for key, value in self.current.items()}
        kwargs = {key: value for key, value in kwargs.items()}
        data.update(kwargs)

        for key, value in kwargs.items():
            if isinstance(value, float) or isinstance(value, int):
                self.all[key].append(value)

        for key, value in data.items():
            self.keys.add(key)
            if isinstance(value, float) or isinstance(value, int):
                if self.use_wandb:
                    wandb.log(
                        {f"average/{key}": value}, step=self.episode
                    )

        self.statistics.append(data)
        self.current = dict()
        self.episode += 1

    def save_hparams(self, hparams):
        """Save hparams to a json file."""
        with open(f"{self.log_dir}/hparams.json", "w") as f:
            json.dump(hparams, f)
        if self.use_wandb:
            wandb.config.update(hparams)

    def export_to_json(self):
        """Save the statistics to a json file."""
        with open(f"{self.log_dir}/statistics.json", "w") as f:
            json.dump(self.statistics, f)
        with open(f"{self.log_dir}/all.json", "w") as f:
            json.dump(self.all, f)

    def load_from_json(self, log_dir=None):
        """Load the statistics from a json file."""
        log_dir = log_dir if log_dir is not None else self.log_dir

        with open(f"{log_dir}/statistics.json", "r") as f:
            self.statistics = json.load(f)
        with open(f"{log_dir}/all.json", "r") as f:
            self.all = json.load(f)
        for key in self.all.keys():
            self.keys.add(key)

    def log_hparams(self, hparams, metrics=None):
        """Log hyper parameters together with a metric dictionary."""
        if self.use_wandb:  # Do not save.
            wandb.config.update(hparams)
        for key, value in metrics.items():
            wandb.run.summary[key] = value

    def delete_directory(self):
        """Delete writer directory.

        Notes
        -----
        Use with caution. This will erase the directory, not the object.
        """
        shutil.rmtree(self.log_dir)

    def change_log_dir(self, new_log_dir):
        """Change log directory."""
        log_dir = new_log_dir
        try:
            self.delete_directory()
        except FileNotFoundError:
            pass
        self.log_dir = safe_make_dir(log_dir)

        try:
            self.load_from_json()  # If json files in log_dir, then load them.
        except FileNotFoundError:
            pass
