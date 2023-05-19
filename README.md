# Reinforcement-Learning-for-Taxi-v3-Environment
This GitHub repository contains code for implementing and evaluating different reinforcement learning algorithms on the Taxi-v3 environment using OpenAI Gym. The project focuses on two approaches: Q-Learning and Q-Network (Deep Q-Network).

The main components of the code include:

1. Environment Setup: The code initializes the Taxi-v3 environment and performs necessary setup steps, such as rendering an initial state.

2. Q-Network: A Q-Network is implemented using TensorFlow and Keras. The network consists of an input layer, a dense hidden layer, and a dense output layer. The model is compiled with the Adam optimizer and Mean Squared Error (MSE) loss function.

3. Evaluation Functions: The project includes evaluation functions for both the Q-Network and Q-Learning approaches. These functions calculate the mean, minimum, maximum, and standard deviation of rewards obtained during evaluation episodes.

4. Q-Learning: The Q-Learning algorithm is implemented using a Q-table. The code performs iterations of the Q-Learning algorithm, updating the Q-values based on the rewards and maximizing future expected rewards.

5. Q-Network Learning and Evaluation: The Q-Network is trained using episodes and batches of one-hot encoded states and corresponding Q-values. The code also evaluates the Q-Network by calculating the mean reward over evaluation episodes.

The repository provides the complete source code, including the necessary imports and function definitions. Additionally, it includes plotting code to visualize the learning progress and performance of the reinforcement learning algorithms.
