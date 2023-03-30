"""Implementation of a Logger class. Adapted from RLLib"""
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import wandb
from numpy import bool_

from utils import WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY


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
    use_wandb: bool, optional.
        Flag that indicates whether to save the results in the wandb.
    """

    def __init__(
        self,
        name,
        comment="",
        filename="sysout.txt",
        safe_log_dir=True,
        log_dir=None,
        save_statistics=False,
        log_episodes=False,
        use_wandb=True,
        offline_mode=False
    ):

        self.statistics = list()
        self.current = dict()
        self.all = defaultdict(list)
        self.keys = set()

        self.episode = 0

        self.use_wandb = use_wandb
        self.save_statistics = save_statistics
        self.log_episodes = log_episodes

        now = datetime.now()
        current_time = now.strftime("%b%d_%H_%M_%S")
        comment = comment + "_" + current_time if len(comment) else current_time
        if log_dir is None:
            log_dir = f"runs/{name}/{comment.replace(' ','_')}"
        if not safe_log_dir:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.log_dir = log_dir
        else:
            self.log_dir = safe_make_dir(log_dir)
        self.file = os.path.join(log_dir, filename)

        if offline_mode:
            os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        if use_wandb:
            wandb.init(
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                name=name,
                notes=comment,
                dir=log_dir,
                mode="offline" if offline_mode else "online"
            )
            wandb.define_metric("test_returns", summary="mean")
            wandb.define_metric("train_returns", summary="mean")

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
            if isinstance(value, bool_):
                value = int(bool(value))

            if key not in self.current:
                self.current[key] = (1, value)
            else:
                count, old_value = self.current[key]
                new_count = count + 1
                new_value = old_value + (value - old_value) * (1 / new_count)
                self.current[key] = (new_count, new_value)

            self.all[key].append(value)

            if self.use_wandb and self.log_episodes:
                wandb.log(
                    {f"episode_{self.episode}/{key}": self.current[key][1]},
                    step=self.current[key][0],
                )

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

    def log_metrics(self, hparams=None, metrics=None):
        """Log hyper parameters together with a metric dictionary."""
        if hparams is not None:
            self.save_hparams(hparams)
        if self.use_wandb:
            wandb.log(metrics)
        if self.save_statistics:
            self.export_to_json()

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

    def write(self, message):
        sys.__stdout__.write(message)
        with open(self.file, 'a') as f:
            f.write(message)

    def flush(self):
        sys.__stdout__.flush()
        with open(self.file, 'a') as f:
            f.flush()
