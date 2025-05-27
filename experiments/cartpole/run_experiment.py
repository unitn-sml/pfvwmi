
"""
Data generation based on the pytorch Q Learning tutorial.

Credits: Adam Paszke, Mark Towers
Source: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

"""

import argparse
from collections import namedtuple, deque
import gymnasium as gym
from itertools import count
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from pysmt.shortcuts import *
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pfvwmi import Verifier
from pfvwmi.encoders import NNEncoder
from pfvwmi.models import DET, FFNN, UniformPrior

def pretty_result(result):
    if not result['is_sat']:
        P = 0
    else:
        P = result['wmi']/result['Z']

    return f"P = {P} ({result['t_total']} s.)"


# QLEARNING
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005


parser = argparse.ArgumentParser()

# DET arguments
parser.add_argument('--det_samples', type=int,
                    help="# DET training samples",
                    default=1000)
parser.add_argument('--det_n_min', type=int,
                    help="DET min. instances in internal nodes",
                    default=500)
parser.add_argument('--det_n_max', type=int,
                    help="DET max. instances in leaves",
                    default=1000)
# NN arguments
parser.add_argument('--hidden', type=int, nargs='+',
                    help="NN: Hidden layer size",
                    #default=[16, 16])
                    default=[4, 4])

parser.add_argument('--num_episodes', type=int,
                    help="# training episodes for the DQN",
                    default=1000)
parser.add_argument('--batch_size', type=int,
                    help="NN: Batch size", default=32)
parser.add_argument('--lr', type=float,
                    help="NN: Learning rate",
                    #default=5e-4)
                    default=1e-3)
# experiment arguments
parser.add_argument('--seed', type=int, help="Seed number",
                    default=666)
parser.add_argument('--k', type=float,
                    help="Check Pr(post|pre) >= k", default=1.0)
parser.add_argument('--partitions', type=str,
                    help="Mode / # partitions (no partitioning)",
                    default='sample-0')
parser.add_argument('--data_folder', type=str,
                    help="Results folder",
                    default="JAIR")    

args = parser.parse_args()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

    
if not os.path.isdir(args.data_folder):
    os.mkdir(args.data_folder)

nn_str = "-".join(map(str, args.hidden)) + f'x{args.num_episodes}' + f'bs{args.batch_size}' + f'lr{args.lr}'
tnet_path = os.path.join(args.data_folder, f'tnet({nn_str}).json')
pnet_path = os.path.join(args.data_folder, f'pnet({nn_str}).json')
episode_durations_path = os.path.join(args.data_folder, f'episode_durations({nn_str}).json')
det_path = os.path.join(args.data_folder, f'det({nn_str}).json')

env = gym.make("CartPole-v1")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

steps_done = 0

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

episode_durations = []
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)

    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')

    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy(), color="blue", alpha=0.666)

    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), color="purple", linestyle="--", alpha=0.666)

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]],
                            device=device, dtype=torch.long)


if not os.path.isfile(tnet_path) or not os.path.isfile(pnet_path):

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset(seed=args.seed)
    n_observations = len(state)

    dimensions = [int(n_observations)] + args.hidden + [int(n_actions)]
    policy_net = FFNN(dimensions, False).to(device)
    target_net = FFNN(dimensions, False).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(),
                            lr=args.lr,
                            amsgrad=True)
    memory = ReplayMemory(10000)

    def optimize_model():

        if len(memory) < args.batch_size:
            return

        transitions = memory.sample(args.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)),
                                      device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(args.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + \
            reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()


    for i_episode in range(args.num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)

            state = next_state
            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)

                target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

    print("DQN training complete")
    policy_net.save(pnet_path)
    target_net.save(tnet_path)
    with open(episode_durations_path, 'wb') as f:
        pickle.dump(episode_durations, f)

else:
    policy_net = FFNN.load(pnet_path)
    target_net = FFNN.load(tnet_path)
    with open(episode_durations_path, 'rb') as f:
        episode_durations = pickle.load(f)


print('Plotting')    
plot_durations(show_result=True)
plt.ioff()
#plt.show()
plot_path = os.path.join(args.data_folder, f'plot({nn_str}).pdf')
plt.savefig(plot_path)
plt.close()

vnames = ['cpos', 'cvel', 'pangle', 'pvel']
feats =[(vname, 'real') for vname in vnames]

loose_bounds = [[-4.8, 4.8],
                [-10000, 10000],
                [-0.418, 0.418],
                [-10000, 10000]]

uniprior = UniformPrior(feats, loose_bounds)

if os.path.isfile(det_path):
    print(f"Found DET at: {det_path} ...", end=" ")
    det = DET.load(det_path)
else:
    samples = []
    det_ntrain = args.det_samples
    det_nvalid = int(det_ntrain * 0.2)
    det_ntest = int(det_ntrain * 0.2)
    det_ntotal = det_ntrain + det_nvalid + det_ntest
    while len(samples) < det_ntotal:
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action = select_action(state)
            observation, _, terminated, truncated, _ = env.step(action.item())

            if terminated or truncated:
                break

            samples.append(observation)

            # Move to the next state
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    samples = np.array(samples)
    print(f"Training DET with nmin: {args.det_n_min}, nmax: {args.det_n_max} ...", end=" ")
    det = DET(feats, # feats
              samples[:det_ntrain],
              n_min=args.det_n_min, n_max=args.det_n_max)
    det.prune(samples[det_ntrain:det_ntrain+det_nvalid])
    det.save(det_path)

# VERIFICATION

results_path = os.path.join(args.data_folder, f'results({nn_str}).json')
if os.path.isfile(results_path):
    print("Found results at:", results_path)
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
else:
    results = {}

smt_x = [Symbol(vname, REAL) for vname in vnames]
smt_y = [Symbol('y_left', REAL), Symbol('y_right', REAL)]
domain = set(smt_x)
cart_position = smt_x[0]
pole_angle = smt_x[2]
priors = {'det' : det,
          'uniform' : uniprior}

cart_left = LT(cart_position, Real(0))
pole_left = LT(pole_angle, Real(0))

action_left = GT(*smt_y)

properties = {
    'cart-leftside' : (Bool(True), cart_left),
    'cart-rightside' : (Bool(True), Not(cart_left)),
    'pole-left' : (Bool(True), pole_left),
    'pole-right' : (Bool(True), Not(pole_left)),
    'go-left' : (Bool(True), action_left),
#    'go-right' : (Bool(True), Not(action_left)),
    'go-left-correct' : (pole_left, action_left),
    'go-right-correct' : (Not(pole_left), Not(action_left)),
}

partition_mode, n_partitions = args.partitions.split('-')
sys_enc = NNEncoder(target_net, smt_x, smt_y,
                    seed=args.seed,
                    n_partitions=int(n_partitions),
                    partition_mode=partition_mode
                    )
    
for prsname in priors:
    verifier = Verifier(sys_enc, priors[prsname], domain,
                        use_ibp=True, use_cache=True)

    for pryname in properties:
        key = (prsname, pryname)        
        if key not in results:

            if prsname == "uniform" and pryname in ['cart-leftside',
                                                    'cart-rightside',
                                                    'pole-left',
                                                    'pole-right']:
                output = {'is_sat' : True,
                          'wmi' : 0.5,
                          'Z' : 1,
                          't_total' : 0.0}
            else:
                pre, post = properties[pryname]
                _ = verifier.check_property(args.k, pre, post)
                output = verifier.last_run_stats

            results[key] = output
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)

            print("\n\n")

        
        print(f"Results[{key}]: {pretty_result(results[key])}\n")
                            



    
terminated_left = 0
n_simulations = 10000
for _ in range(n_simulations):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    for t in count():
        action = select_action(state)
        observation, _, terminated, truncated, _ = env.step(action.item())

        if terminated or truncated:
            if terminated and observation[3] > 0.4:
                terminated_left += 1
            break

        # Move to the next state
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

print(f"\n\nTerminated left: {terminated_left}/{n_simulations} = {terminated_left/n_simulations}")
