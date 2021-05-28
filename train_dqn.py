import time
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import deque

from dqn_agent import DQNAgent
from banana_env import BananaEnv


def parse_args():
    """
    Parse specified arguments from command line

    Args:
        None

    Returns:
        Argparse NameSpace object containing command line arguments

    """
    parser = argparse.ArgumentParser(description='DQN hyperparameters')
    parser.add_argument('--agent-name', type=str, default='dqn-agent',
        help='DQN agent name'
    )
    parser.add_argument('--buffer-size', type=int, default=int(1e5),
        help='Experience replay buffer size'
    )
    parser.add_argument('--batch-size', type=int, default=64,
        help='Mini-batch size'
    )
    parser.add_argument('--env-name', type=str, default='banana-env',
        help='BananaEnv name'
    )
    parser.add_argument('--eps-start', type=float, default=1.0,
        help='Starting value of epsilon, for epsilon-greedy action selection'
    )
    parser.add_argument('--eps-end', type=float, default=0.01,
        help='Minimum value of epsilon'
    )
    parser.add_argument('--eps-decay', type=float, default=0.995,
        help='Multiplicative factor (per episode) for decreasing epsilon'
    )
    parser.add_argument('--gammma', type=float, default=0.99,
        help='Discount factor'
    )
    parser.add_argument('--hidden', nargs='*', default=[256,128,64],
        help='Model hidden layer sizes'
    )
    parser.add_argument('--lr', type=float, default=5e-4, help='Learn rate')
    parser.add_argument('--max-t', type=int, default=300,
        help='Maximum number of timesteps per episode'
    )
    parser.add_argument('--n-episodes', type=int, default=2000,
        help='Maximum number of training episodes'
    )
    parser.add_argument('--output', type=str, default='./output',
        help='Directory to save models, logs, & other output'
    )
    parser.add_argument('--run-id', type=int,
        help='Execution run identifier'
    )
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for repeatability'
    )
    parser.add_argument('--sim', type=str,
        default='Banana_Windows_x86_64/Banana.exe',
        help='Path to Unity Banana simulation'
    )
    parser.add_argument('--tau', type=float, default=1e-3,
        help='Interpolation weight for soft update of target parameters'
    )
    parser.add_argument('--test', action='store_true',
        help='Test mode, no agent training'
    )
    parser.add_argument('--update-every', type=int, default=4,
        help='Update frequency'
    )
    parser.add_argument('--verbose', action='store_true', help='Verbosity')
    args = parser.parse_args()

    # Convert hidden layer argument to list of ints
    args.hidden = [int(size) for size in args.hidden]

    # Convert current time to run ID if none specified
    if args.run_id is None:
        args.run_id = int(time.time())

    # Convert output directory to path object including run ID
    args.output = Path(args.output + '/' + str(args.run_id) + '/')

    return args

def plot_performance(scores, name):
    """
    Plot summary of DQN performance on environment

    Args:
        scores (list of float): Score per simulation episode
        name (Path): Name for file

    """

    # Create avg score
    avg_score = [np.mean(scores[:i+1]) for i in range(len(scores))]

    # Plot scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(np.arange(len(scores)), scores, color='cyan', label='Scores')
    plt.plot(avg_score, color='blue', label='Average')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(loc='upper left')
    plt.show(block=False)

    # Save figure
    name.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(name)
    plt.close()

    return

def write2path(text, path):
    """
    Write text to path object

    Args:
        text (str): Text to log
        path (Path): Path object
    
    Returns:
        None
    
    """

    # Create path
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write text to path
    if not path.exists():
        path.write_text(text)
    else:
        with path.open('a') as f:
            f.write(text)
    
    return

def eval_agent(agent, env, eval_type, **kwargs):
    """
    Evaluate agent using environment

    Args:
        agent (DQNAgent): Deep q-learning agent
        env (BananaEnv): Unity banana simulation environment
        eval_type (str): Training or testing agent

    Returns:
        None

    """
    scores = []                                 # scores from each episode
    score_goal = 13.0                           # goal to get to
    window_size = 100                           # size for rolling window
    eval_options = ['train', 'test']              # evaluation options
    scores_window = deque(maxlen=window_size)   # last 100 scores

    # Error check
    if eval_type.lower() not in eval_options:
        raise ValueError(
            'Invalid eval_type specified. Options are {}'.format(eval_options)
        )

    # Initialize key word argument variables
    n_episodes = kwargs.get('n_episodes', 2000)
    max_t = kwargs.get('max_t', 300)
    eps_start = kwargs.get('eps_start', 1.0)
    eps_end = kwargs.get('eps_end', 0.01)
    eps_decay = kwargs.get('eps_decay', 0.995)
    run_id = kwargs.get('run_id', int(time.time()))
    output = kwargs.get('output', Path('./output/' + str(run_id) + '/'))
    verbose = kwargs.get('verbose', False)

    # Initialize epsilon
    eps = eps_start

    # Create log name
    prefix = str(run_id) + '__' + agent.name + '__' + env.name
    log = output / (prefix + '__performance.log')

    # Train for n_episodes
    for i_episode in range(1, n_episodes+1):

        # Reset environment
        score = 0
        state = env.reset()

        # Learn for max_t steps
        for t in range(max_t):

            # Get action from agent using current policy
            if eval_type == 'train':
                action = agent.act(state, eps)
            else:
                action = agent.act(state, train=False)

            # Perform action in environment
            state, action, reward, next_state, done = env.step(action)

            # Update agent
            if eval_type == 'train':
                agent.step(state, action, reward, next_state, done)

            # Update next state & score
            state = next_state
            score += reward

            # Check terminal condition
            if done:
                break
        
        # Save most recent scores
        scores_window.append(score)
        scores.append(score)

        # Decrease epsilon
        eps = max(eps_end, eps_decay*eps)

        # Print & log episode performance
        window_summary = '\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_window))
        print(window_summary, end="")
        if eval_type == 'test':
            write2path(window_summary, log)

        # Print & log performance of last window_size runs
        if i_episode % window_size == 0:
            window_summary = '\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window))
            print(window_summary)
            write2path(window_summary, log)

        # Terminal condition check
        if np.mean(scores_window) >= score_goal:
            window_summary = (
                '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'
            ).format(i_episode-100, np.mean(scores_window))
            print(window_summary)
            write2path(window_summary, log)
            break

    # Save model 
    output.mkdir(parents=True, exist_ok=True)
    torch.save(agent.qnetwork_local.state_dict(), output / (prefix + '__model.pth'))
    
    # Plot training performance
    if eval_type == 'train':
        plot_performance(scores, output / (prefix + '__training.png'))
    else:
        plot_performance(scores, output / (prefix + '__testing.png'))

    # Save evaluation parameters
    parameters = {
        'n_episodes': n_episodes,
        'max_t': max_t,
        'eps_start': eps_start,
        'eps_end': eps_end,
        'eps_decay': eps_decay
    }
    with open(output / (prefix + '__parameters.json'), 'w') as file:
        json.dump(parameters, file, indent=4, sort_keys=True)

    return

def main(args):
    """
    Train DQN agent

    Args:
        args: Argparse NameSpace object containing command line arguments

    Returns:
        None

    """

    # Create environment
    train_mode = False if args.test else True
    env = BananaEnv(args.sim, name=args.env_name, train=train_mode, seed=args.seed, verbose=args.verbose)

    # Create agent
    agent = DQNAgent(env.state_size, env.action_size, name=args.agent_name, **vars(args))

    # Evaluate agent
    train_mode = 'test' if args.test else 'train'
    eval_agent(agent, env, train_mode, **vars(args))

    # Close env
    env.close()

    return


if __name__ == "__main__":
    """
    Execute script
    """
    args = parse_args()
    main(args)
