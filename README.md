# Probabilistic Model-based Meta Reinforcement Learning

This repo contains the source code to reproduce the experiments in the paper 
["Data-Efficient Task Generalization via Probabilistic Model-based Meta Reinforcement Learning"](https://ieeexplore.ieee.org/document/10452796).
Here we provide instructions on how to set up and run the code to reproduce the experiments reported in the paper.
Users can also create new environments and use custom datasets using the example code provided here.

### Citation

Arjun Bhardwaj, Jonas Rothfuss, Bhavya Sukhija, Yarden As, Marco Hutter, Stelian Coros, and Andreas Krause. 
"Data-Efficient Task Generalization via Probabilistic Model-Based Meta Reinforcement Learning." In 
_IEEE Robotics and Automation Letters (RA-L)_, vol. 9, no. 4, pp. 3918-3925, April 2024.

```
@ARTICLE{10452796,
  author={Bhardwaj, Arjun and Rothfuss, Jonas and Sukhija, Bhavya and As, Yarden and Hutter, Marco and Coros, Stelian and Krause, Andreas},
  journal={IEEE Robotics and Automation Letters}, 
  title={Data-Efficient Task Generalization via Probabilistic Model-Based Meta Reinforcement Learning}, 
  year={2024},
  volume={9},
  number={4},
  pages={3918-3925},
  keywords={Task analysis;Adaptation models;Metalearning;Uncertainty;Bayes methods;Robots;Artificial neural networks;Reinforcement learning;model learning for control;learning from experience},
  doi={10.1109/LRA.2024.3371260}}
```

## Setup

### Setting up the conda environment
We recommend using [Anaconda](https://www.anaconda.com/) to set up an environment with the dependencies of this repository. After installing Anaconda, the following commands set up the environment:
1. Create a new conda environment:
    ```bash
    conda create -n meta_rl python=3.7
    ```
2. Activate the environment and install the required packages:
    ```bash
    conda activate meta_rl
    pip install -e .
    ```

### Installing MuJoCo

Download MuJoCo Pro 1.31 binaries from [Roboti website](https://www.roboti.us/download.html) along with the [license key](https://www.roboti.us/file/mjkey.txt) and save them in the directory `~/.mujoco/`. Detailed installation instructions can be found on the MuJoCo Python [project page](https://github.com/openai/mujoco-py/tree/0.5.7#obtaining-the-binaries-and-license-key). After installing MuJoCo, set the correct environment paths for your dynamic loaders: 
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro131/bin:/usr/lib/nvidia
```
You can add these lines to your `.bashrc` or `.bash_profile` file to set the environment variables automatically.

## Reproducing the experiments

The `experiments/launch_experiments.py` script can be used to run the experiments for the different algorithms and environments. 
The script takes the following arguments:

- `env-config-file`: The name of the environment configuration file. 
The environment configuration file is a yaml file that contains the environment name, training parameters, and agent parameters used for that environment. 
For a complete list of agent and environment parameters, see `experiments/meta_rl_experiments/parser.py`.
The environment configuration files for the different environments are provided in the `experiments/configs/env_configs` directory.

- `agent-name`: The agent to be used for training. This code base provides the implementation of the following model-based meta-RL and model-based RL algorithms:

  - [PACOH-RL](https://arxiv.org/abs/2311.07558): "pacoh_rl"
  - [PACOH-RL (Greedy)](https://arxiv.org/abs/2311.07558): "pacoh_rl_greedy"
  - [PACOH-RL (SAC)](https://arxiv.org/abs/2311.07558): "pacoh_rl_sac"
  - [GrBAL](https://arxiv.org/abs/1803.11347): "grbal"
  - [H-UCRL](https://arxiv.org/abs/2006.08684): "hucrl"
  - [PETS-DS](https://arxiv.org/abs/1805.12114): "pets"
  - [MBPO](https://arxiv.org/abs/1906.08253): "mbpo"

- `collect-meta-data`: If set to `True`, the script will collect meta-training data for the agent using the specified environment and save it to the `meta_rl_experiments` directory.
If the meta-training data has been collected previously, set this to `False` to use the existing data.
- `num-seeds`: The number of random seeds to use for training the agent. This will run `num-seeds` number of experiments for the given agent and environment.
- `exp-name`: The name to be used for logging the experiment.
- `use-wandb`: If set to `True`, the script will log the training progress to [Weights and Biases](https://wandb.ai/).

## Extending the code

The code can be used with custom environments and datasets. The following sections provide instructions on how to create new environments and use custom datasets.

### Creating new environments
The code base provides a template for creating new environments. The `lib/environments/custom_envs/random_template_env.py` file contains the template for creating a new random environment.
To train an agent with a new environment, follow these steps:
- Add the new environment file to the `lib/environments/custom_envs` directory. The new environment file should implement all the methods required by the `RandomTemplateEnv` class in `random_template_env.py`.
- Add the new environment to the `lib/environments/custom_envs/__init__.py` file.
- Add a configuration file for the new environment in the `experiments/configs/env_configs` directory with the parameter `name` set to the name of the environment class.
- Run the `launch_experiments.py` script with the new environment configuration file and argument `group-name` set to "custom_envs".
For more details, see `utils/get_environments.py`.

### Using custom datasets
Users can also use custom meta-training datasets with the provided algorithms. 
The file `lib/datasets/dataloaders/dataloader.py` contains a demo script for loading and saving a new dataset. Users can
implement their own dataloaders, which are then stored in the appropriate format and can be used later for meta-learning.
