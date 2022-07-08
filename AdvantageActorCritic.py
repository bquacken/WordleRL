import tensorflow as tf
from tensorflow import keras
from keras import layers

import numpy as np

critic_loss_weight = 0.5
actor_loss_weight = 1
entropy_loss_weight = 0.01

class AdvantageActorCritic(keras.Model):
    """
    A2C Model for Wordle.
    Input: State of the Wordle Game, Shape (443)
    Output: Logits with shape as the length of the number of actions
    
    mx: (n, 130) Sparse Matrix of one-hot encoded words that are the possible actions. 
    actions_list: list of indices for where corresponding actions can be found in the total_words list, used for training on smaller 100/1000 word games
    """
    def __init__(self, mx, actions_list):
        super().__init__()
        self.mx = mx
        self.init = keras.initializers.random_uniform()
        self.actions_list = actions_list
        self.n_outputs = 130
        self.leaky = layers.ReLU()
        self.dense1 = layers.Dense(256, activation= self.leaky)
        self.dense2 = layers.Dense(256, activation= self.leaky)
        self.dense3 = layers.Dense(256, activation= self.leaky)
        self.policy1 = layers.Dense(self.n_outputs)
        self.value1 = layers.Dense(256, activation = self.leaky)
        self.value = layers.Dense(1)
        
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        policy = self.policy1(x)
        policy = tf.linalg.matmul(self.mx, policy, transpose_b=True)
        policy_logits = tf.linalg.matrix_transpose(policy)
        value = self.value1(x)
        value = self.value(value)
        return value, policy_logits
    
    def action_value(self, state):
        value, logits = self.call(state)
        action = tf.random.categorical(logits, 1)
        return action, value
    
def critic_loss(rewards, predicted_values):
    huber = keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    return huber(rewards, predicted_values)*critic_loss_weight

def actor_loss(combined, policy_logits):
    actions = combined[:, 0]
    advantages = combined[:, 1]
    sparse_ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                           reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    
    actions = tf.cast(actions, tf.int32)
    policy_loss = sparse_ce(actions, policy_logits, sample_weight=advantages)
    
    probs = tf.nn.softmax(policy_logits)
    entropy_loss = keras.losses.categorical_crossentropy(probs, probs)
    
    return policy_loss*actor_loss_weight - entropy_loss*entropy_loss_weight

#Compute Advantages and Discounted Rewards
def compute_advantages(rewards, values, dones):
    discount_factor = 0.99
    advantages = np.zeros(len(rewards))
    advantages[-1] = rewards[-1] - values[-1]
    out_rewards = np.zeros(len(rewards))
    out_rewards[-1] = rewards[-1]
    n = len(rewards)
    for i in reversed(range(len(rewards)-1)):
        out_rewards[i] = rewards[i] + (1 - dones[i])*discount_factor*out_rewards[i+1]

    advantages = out_rewards - values
    return out_rewards, advantages
