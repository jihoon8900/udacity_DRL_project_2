import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
NUM_LEARN = 10          # number of learning 
NUM_TIME_STEP = 20      # every NUM_TIME_STEP do update
EPSILON = 1.0           # epsilon to noise of action
EPSILON_DECAY = 2e-6    # epsilon decay to noise epsilon of action
POLICY_DELAY = 3        # delay for policy update (TD3)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, env, random_seed=15, device=device):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.env = env
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]
        self.device = device

        self.state_size = self.brain.vector_observation_space_size
        self.action_size = self.brain.vector_action_space_size
        self.seed = random.seed(random_seed)

        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY

        self.iteration = 0

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.state_size, self.action_size, random_seed, fc1_units=256, fc2_units=128).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, random_seed, fc1_units=256, fc2_units=128).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local_1 = Critic(self.state_size, self.action_size, random_seed, fcs1_units=256, fc2_units=128).to(device)
        self.critic_target_1 = Critic(self.state_size, self.action_size, random_seed, fcs1_units=256, fc2_units=128).to(device)
        self.critic_optimizer_1 = optim.Adam(self.critic_local_1.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.critic_local_2 = Critic(self.state_size, self.action_size, random_seed, fcs1_units=256, fc2_units=128).to(device)
        self.critic_target_2 = Critic(self.state_size, self.action_size, random_seed, fcs1_units=256, fc2_units=128).to(device)
        self.critic_optimizer_2 = optim.Adam(self.critic_local_2.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(self.action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done, t):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and t % NUM_TIME_STEP == 0:
            for _ in range(NUM_LEARN):
                self.iteration += 1
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.epsilon * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        actions_next += torch.clip(torch.normal(mean=0., std=0.2, size=actions_next.shape), -1.0, 1.0).to(self.device) ##################### TD3
        Q_targets_next = torch.min(self.critic_target_1(next_states, actions_next), self.critic_target_2(next_states, actions_next)) ##################### TD3

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected_1 = self.critic_local_1(states, actions)
        Q_expected_2 = self.critic_local_2(states, actions)
        critic_loss_1 = F.mse_loss(Q_expected_1, Q_targets.detach())
        critic_loss_2 = F.mse_loss(Q_expected_2, Q_targets.detach())

        # Minimize the loss
        self.critic_optimizer_1.zero_grad()
        self.critic_optimizer_2.zero_grad()
        critic_loss_1.backward()
        critic_loss_2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local_1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.critic_local_2.parameters(), 1.0)
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.step()

        if self.iteration % POLICY_DELAY == 0:
            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local_1(states, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1.0)
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local_1, self.critic_target_1, TAU)
            self.soft_update(self.critic_local_2, self.critic_target_2, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)                 
        
        self.epsilon -= self.epsilon_decay    

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def train(self, n_episodes=1800, max_t=3000):
        """
        DDPG.    
        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        self.scores = []                                                # list containing scores from each episode
        scores_window = deque(maxlen=100)                               # last 100 scores
        for i_episode in range(1, n_episodes+1):
            env_info = self.env.reset(train_mode=True)[self.brain_name] # reset the environment
            state = env_info.vector_observations                        # get the current state
            score = 0
            self.noise.reset()                                          # reset OU noise
            for t in range(max_t):
                action = self.act(state, add_noise=True)
                env_info = self.env.step(action)[self.brain_name]       # send the action to the environment
                next_state, reward, done = env_info.vector_observations, env_info.rewards, env_info.local_done
                self.step(state, action, reward, next_state, done, t)
                state = next_state
                score += np.mean(reward)
                if any(done):
                    break 
            
            self.iteration = 0                                          # reset iteration
            scores_window.append(score)                                 # save most recent score
            self.scores.append(score)                                   # save most recent score
            print('\rEpisode {}\tAverage Score : {:.2f} \t eps : {:.3f}'.format(i_episode, np.mean(scores_window), self.epsilon), end="")
            if i_episode % 10 == 0:
                print('\rEpisode {}\tAverage Score : {:.2f} \t eps : {:.3f}'.format(i_episode, np.mean(scores_window), self.epsilon))
            if np.mean(scores_window) >= 30.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                torch.save(self.actor_local.state_dict(), 'saved_TD3_actor.pth')
                torch.save(self.critic_local_1.state_dict(), 'saved_TD3_critic_1.pth')
                torch.save(self.critic_local_2.state_dict(), 'saved_TD3_critic_2.pth')
                break
        return self.scores

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        for idx in range(len(state)):
            e = self.experience(state[idx], action[idx], reward[idx], next_state[idx], done[idx])
            self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)