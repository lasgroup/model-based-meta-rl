from setuptools import setup, find_packages

# Function to read the requirements.txt file
def parse_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()

# Read the dependencies from requirements.txt
requirements = parse_requirements('requirements.txt')

git_dependencies = [
    "rllib@git+https://github.com/arbhardwaj98/rllib.git#egg=rllib",
    "rand_param_envs@git+https://github.com/dennisl88/rand_param_envs.git#egg=rand_param_envs"
]

# Define the setup configuration
setup(
    name='meta_mbrl',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements + git_dependencies,
    author='Arjun Bhardwaj',
    author_email='abhardwaj@ethz.ch',
    description='A repository for Model-Based Reinforcement Learning and Meta-Reinforcement Learning Algorithms',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7<3.8',
)
