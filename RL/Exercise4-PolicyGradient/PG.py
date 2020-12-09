import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import os


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class PG:

    def __init__(self, state_size, action_size, render=False,
                 load_model=False, gamma=0.99, learning_rate=0.001):
        # 设置环境参数
        self.state_size = state_size
        self.action_size = action_size
        # 回报衰减率
        self.gamma = gamma
        # 学习速率
        self.learning_rate = learning_rate
        # 一个episode的观测值、动作值、回报值
        self.states, self.gradients, self.rewards, self.probs = [], [], [],[]
        # 创建策略网络
        self.model = self.build_model()
        # 记录损失值
        self.history = LossHistory()
        self.losses_list = []
        # 是否开启渲染
        self.render = render
        # 是否加载模型
        self.load_model = load_model

    def build_model(self):
        # 创建一个激活函数为'tanh'的隐藏层，'softmax'的输出层
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(units=10, input_dim=self.state_size, activation='tanh',
                                     kernel_initializer='random_uniform'))
        model.add(keras.layers.Dense(units=self.action_size, activation='softmax',
                                     kernel_initializer='random_uniform'))
        model.summary()
        model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def memorize(self, state, action, reward, prob):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def choose_action(self, state):
        # 根据网络输出的概率选择动作
        state = state.reshape([1, state.shape[0]])
        prob = self.model.predict(state, batch_size=1).ravel()
        self.probs.append(prob)
        action = np.random.choice(range(self.action_size), 1, p=prob)
        return np.squeeze(action), prob

    def greedy_action(self, state):
        # 采用贪婪策略选择动作
        state = state.reshape([1, state.shape[0]])
        prob = self.model.predict(state, batch_size=1)
        return np.argmax(prob[0])

    def discount_rewards(self):
        # 一个episode完成后，反推每个状态的累计回报
        discount_rewards = np.zeros_like(self.rewards)
        running_add = 0
        for i in reversed(range(0, len(self.rewards))):
            running_add = running_add * self.gamma + self.rewards[i]
            discount_rewards[i] = running_add
        discount_rewards = (discount_rewards - np.mean(discount_rewards))/np.std(discount_rewards)
        return discount_rewards

    def train(self):
        rewards = self.discount_rewards()
        rewards = np.vstack(rewards)
        gradients = self.gradients * rewards    # (m,2) * (m,1)
        X = np.squeeze(self.states)     # (m,2)
        Y = self.probs + self.learning_rate * gradients     # (m,2) + c * (m,2)*(m,1)
        self.model.fit(X, Y, epochs=1, verbose=0, callbacks=[self.history])
        self.losses_list.append(self.history.losses[0])
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []


