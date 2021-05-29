[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: ./images/agent_performance.png "Agent Performance"

# Project 1: Navigation

## Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:

- **`0`** - move forward
- **`1`** - move backward
- **`2`** - turn left
- **`3`** - turn right

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Installation

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

   - __Linux__ or __Mac__:
  
    ```bash
    conda create --name navigation python=3.6
    source activate navigation
    ```

   - __Windows__:

    ```bash
    conda create --name navigation python=3.6 
    activate navigation
    ```

2. Install [dependencies](#dependencies)

3. Download [Unity Simulation Environment](#unity-simulation-environment)

### Dependencies

To install required dependencies to execute code in the repository, follow the instructions below.

1. Install [PyTorch](https://pytorch.org/)

    ```bash
    conda install pytorch cudatoolkit=10.2 -c pytorch
    ```

    A fresh installation of [PyTorch](https://pytorch.org/) is recommended due to installation errors encountered when installing the [Udacity Deep Reinforcement Learning repository](https://github.com/udacity/deep-reinforcement-learning), such as [the following Windows error](https://github.com/udacity/deep-reinforcement-learning/issues/13), as well as outdated driver issues when using `torch==0.4`.

2. Install required packages using `pip` from main repository directory

    ```bash
    conda install pytorch cudatoolkit=10.2 -c pytorch
    ```

### Unity Simulation Environment

1. Download the environment from one of the links below.  You need only select the environment that matches your
   operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your
    computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Save the file locally and unzip (or decompress) the file.

## Instructions

The `train_dqn.py` script can be used to train or evaluate an agent. Logs of agent parameters, agent performance (as shown below), and environment evaluation settings are saved during script execution. Agent and associated model hyperparameters, such as model architecture, are configurable via command line arguments. See [help](#help) section for more details about available parameters.

![Agent Performance][image2]

### Training an Agent

Training an agent only requires specifying the path to the [downloaded Unity simulation environment](#getting-started)

```bash
python -m navigation.train_dqn --sim Banana_Windows_x86_64/Banana.exe
```

Model training parameters are configurable via command line. Certain variables such as agent and env name are used for
logging of agent parameters and environment performance results.

```bash
python -m navigation.train_dqn --n-episodes 500 --batch-size 128 --hidden 64 64 --agent-name dqn --env-name train-env --simBanana_Windows_x86_64/Banana.exe --seed 5 --verbose
```

### Continuing Training

To continue training using a previously trained model, specify the path to the previously saved model using the `--load` command line argument.

**NOTE:** Saved model hidden layer sizes must be known and specified using the `--hidden` command line argument. If unknown, search agent parameters JSON for hidden layer sizes specified by either `"qnetwork_local"` or `"qnetwork_target"` keys.

```bash
python -m navigation.train_dqn --load output/1622187609/1622187609__dqn__train-env__model.pth --hidden 64 64 --sim Banana_Windows_x86_64/Banana.exe
```

### Evaluating a Trained Agent

Evaluating a trained agent requires using the `--load` and `--test` arguments simultaneously. The number of evaluation episodes is specified using `--n-episodes` argument, while `--max-t` argument specifies the number of maximum simulation time steps per episode. The maxmimum number of time steps for the Unity banana simulation appears to be 300 (determined empirically).

```bash
python -m navigation.train_dqn --load output/1622187609/1622187609__dqn__train-env__model.pth --hidden 64 64 --sim Banana_Windows_x86_64/Banana.exe --n-episodes 100 --max-t 300 --test
```

### Help

For a full list of available parameters try

```bash
python -m navigation.train_dqn --help

usage: train_dqn.py [-h] [--agent-name AGENT_NAME] [--buffer-size BUFFER_SIZE]
                    [--batch-size BATCH_SIZE] [--env-name ENV_NAME]
                    [--eps-start EPS_START] [--eps-end EPS_END]
                    [--eps-decay EPS_DECAY] [--gammma GAMMMA]
                    [--hidden [HIDDEN [HIDDEN ...]]] [--load LOAD] [--lr LR]
                    [--max-t MAX_T] [--n-episodes N_EPISODES]
                    [--output OUTPUT] [--run-id RUN_ID] [--seed SEED]
                    [--sim SIM] [--tau TAU] [--test]
                    [--update-every UPDATE_EVERY] [--verbose]

DQN hyperparameters

optional arguments:
  -h, --help            show this help message and exit
  --agent-name AGENT_NAME
                        DQN agent name
  --buffer-size BUFFER_SIZE
                        Experience replay buffer size
  --batch-size BATCH_SIZE
                        Mini-batch size
  --env-name ENV_NAME   BananaEnv name
  --eps-start EPS_START
                        Starting value of epsilon, for epsilon-greedy action
                        selection
  --eps-end EPS_END     Minimum value of epsilon
  --eps-decay EPS_DECAY
                        Multiplicative factor (per episode) for decreasing
                        epsilon
  --gammma GAMMMA       Discount factor
  --hidden [HIDDEN [HIDDEN ...]]
                        Model hidden layer sizes
  --load LOAD           Path to model to load
  --lr LR               Learn rate
  --max-t MAX_T         Maximum number of timesteps per episode
  --n-episodes N_EPISODES
                        Maximum number of training episodes
  --output OUTPUT       Directory to save models, logs, & other output
  --run-id RUN_ID       Execution run identifier
  --seed SEED           Seed for repeatability
  --sim SIM             Path to Unity Banana simulation
  --tau TAU             Interpolation weight for soft update of target
                        parameters
  --test                Test mode, no agent training
  --update-every UPDATE_EVERY
                        Update frequency
  --verbose             Verbosity
  ```
