import torch
import argparse
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

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
    parser.add_argument('--buffer-size', type=int, default=int(1e5), 
        help='Experience replay buffer size'
    )
    parser.add_argument('--batch-size', type=int, default=64, 
        help='Mini-batch size'
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
    parser.add_argument('--seed', type=int, default=0, 
        help='Seed for repeatability'
    )
    parser.add_argument('--tau', type=float, default=1e-3, 
        help='Interpolation weight for soft update of target parameters'
    )
    parser.add_argument('--update-every', type=int, default=4, 
        help='Update frequency'
    )
    parser.add_argument('--verbose', action='store_true', help='Verbosity')
    args = parser.parse_args()

    # Convert hidden layer argument to list of ints
    args.hidden = [int(size) for size in args.hidden]

    return args

def dqn(env, agent, n_episodes, max_t, eps_start, eps_end, eps_decay, 
        verbose=False):
    """
    Deep Q-Learning
    
    Args:
        env (BananaEnv): Unity banana simulation environment
        agent (DQNAgent): Deep q-learning agent
        n_episodes (int): Maximum number of training episodes
        max_t (int): Maximum number of timesteps per episode
        eps_start (float): Starting value of epsilon, for epsilon-greedy action 
                           selection
        eps_end (float): Minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing 
                           epsilon
        verbose (boolean): Verbosity

    Returns:
        scores (type): Description
    
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon

    # Train for n_episodes
    for i_episode in range(1, n_episodes+1):

        # Reset environment
        score = 0
        state = env.reset()
        
        # Learn for max_t steps 
        for t in range(max_t):

            # Get action from agent using current policy
            action = agent.act(state, eps)

            # Perform action in environment
            state, action, reward, next_state, done = env.step(action)

            # Update agent 
            agent.step(state, action, reward, next_state, done)

            # Update next state & score
            state = next_state
            score += reward

            # Check terminal condition
            if done:
                break 
        
        # Update scores
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score

        # Decrease epsilon
        eps = max(eps_end, eps_decay*eps)

        # Print scores info
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))

        # In order to solve the environment, your agent must get an average 
        # score of +13 over 100 consecutive episodes
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!'
                '\tAverage Score: {:.2f}'.format(
                    i_episode-100, np.mean(scores_window)
                )
            )
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    
    return scores

def main(args):
    """
    Train DQN agent

    Args:
        args: Argparse NameSpace object containing command line arguments

    Returns:
        None
    
    """

    # Create environment
    env = BananaEnv(train=True, seed=args.seed, verbose=args.verbose)

    # Create agent
    agent = DQNAgent(
        state_size=env.state_size, 
        action_size=env.action_size, 
        **vars(args)
    )

    # Perform deep q-learning
    scores = dqn(
        env=env, 
        agent=agent,
        n_episodes=args.n_episodes, 
        max_t=args.max_t, 
        eps_start=args.eps_start, 
        eps_end=args.eps_end, 
        eps_decay=args.eps_decay,
        verbose=args.verbose
    )

    # Close environment
    env.close()
    
    # Plot scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    return


if __name__ == "__main__":
    """
    Execute script 
    """
    args = parse_args()
    main(args)
