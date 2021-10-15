[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Report for Project 2: Continuous Control for 20 Agents
    
## Introduction

In this `Report.md`, you can see an implementation for the second project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

![Trained Agent][image1]
## Implementation

### 1. Summary

I implement the [Twin Delayed Deep Deterministic policy gradient algorithm (TD3)](https://arxiv.org/abs/1802.09477)

------------
### 2. Details

#### 2-1. Concept

#### 2-2. Networks

##### 2-2-1. Actor

| Layer | Type | Dimension |
|:---:|:---:|:---:|
| `Input` | Input (state) | 33 |
| `BatchNorm` | Batch Normalization | - |
| `1st hidden layer` | Fully connected layer | 256 |
| `BatchNorm` | Batch Normalization | - |
| `Activation function` | LeakyReLu function | - |
| `2nd hidden layer` | Fully connected layer | 128 |
| `BatchNorm` | Batch Normalization | - |
| `Activation function` | LeakyReluReLu function | - |
| `3rd hidden layer` | Fully connected layer | 1 |
| `Activation function` | tanh function | - |
| `(Output)` | Output (action) | 1 |


##### 2-2-2. Critic

| Layer | Type | Dimension |
|:---:|:---:|:---:|
| `Input` | Input (state) | 33 |
| `BatchNorm` | Batch Normalization | - |
| `1st hidden layer` | Fully connected layer | 256 |
| `Activation function` | ReLu function | - |
| `Concat` | Concatenate with action | (257) |
| `2nd hidden layer` | Fully connected layer | 128 |
| `Activation function` | ReLu function | - |
| `3rd hidden layer` | Fully connected layer | 1 |


#### 2-3. Hyperparameters

| parameter    | value  | description                                                                   |
|--------------|--------|-------------------------------------------------------------------------------|
| BUFFER_SIZE  | 1e6    | Number of experiences to keep on the replay memory for the TD3                |
| BATCH_SIZE   | 128    | Minibatch size used at each learning step                                     |
| GAMMA        | 0.99   | Discount applied to future rewards                                            |
| TAU          | 0.001  | Scaling parameter applied to soft update                                      |
| LR_ACTOR     | 0.001  | Learning rate for actor used for the Adam optimizer                           |
| LR_CRITIC    | 0.001  | Learning rate for critic used for the Adam optimizer                          |
| NUM_LEARN    | 10     | Number of learning at each step                                               |
| NUM_TIME_STEP| 20     | Every NUM_TIME_STEP do update                                                 |
| EPSILON      | 4      | Epsilon to noise of action                                                    |
| EPSILON_DECAY| 2e-6   | Epsilon decay to noise epsilon of action                                      |
| POLICY_DELAY | 3      | Delay for policy update (TD3)                                                 |
| n_episodes                    | 3000             | Maximum number of training episodes (Training hyperparameters for `train` function)                                    |
| max_t                         | 3000             | Maximum number of steps per episode (Training hyperparameters for `train` function)                                     |


-----------

### 3. Result and Future works

#### 3-1. Reward

![Reward](https://user-images.githubusercontent.com/53895034/137493903-e5490f93-d98d-44b1-989e-475e69a6bebb.png)


| axis     | value   |
|:--------:|:-------:|
| x-axis   | episode | 
| y-axis   | reward  | 

Environment solved in 162 episodes. You can see relatively stable learning since TD3 is the improved version of DDPG.

#### 3-2. Future works

- Optimizing the hyperparameters
- Deepening the network model 
- Implement with other algorithms (**PPO(Proximal Policy Optimization)**, **SAC(Soft Actor Critic)**)