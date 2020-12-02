import os
import random
import numpy as np
from tensorflow import keras
from collections import deque
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class Agent:

    def __init__(self, state_size,action_size, render=False,
                 load_model=False, gamma=0.99, learning_rate=0.001,
                 epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, batch_size=64,
                 train_start=100, memory_size=2000):
        # 设置环境参数
        self.state_size = state_size
        self.action_size = action_size
        # 是否开启渲染
        self.render = render
        # 是否从文件总加载model
        self.load_model = load_model
        # 设置DQN模型的参数
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # 开启训练神经网络前迭代的次数
        self.train_start = train_start
        # 记忆储存
        self.memory = Memory(memory_size, batch_size)
        # 初始化模型
        self.model = self.build_model()
        self.target_model = self.build_model()
        # 记录损失值
        self.history = LossHistory()
        self.losses_lsit = []

        # 更新TD目标网络步长
        self.step = 0
        self.update_fre = 300

    def build_model(self):
        # 模型创建参考网址：https://tensorflow.google.cn/guide/keras/sequential_model?hl=zh-cn#when_to_use_a_sequential_model
        # 定义3层顺序模型
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(units=128, input_dim=self.state_size, activation='sigmoid',
                                     kernel_initializer='he_uniform'))
        model.add(keras.layers.Dense(units=128, activation='sigmoid', kernel_initializer='he_uniform'))
        model.add(keras.layers.Dense(units=self.action_size, activation='linear', kernel_initializer='he_uniform'))
        # 显示模型信息
        model.summary()
        # 配置训练模型
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
        return model

    def choose_action(self, state):
        # epsilon-greedy
        # 如果概率小于epsilon，则随机选择行为；随着epsilon减小，逐渐倾向选择最优策略
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_values = self.model.predict(state)  # (1,2)
            return np.argmax(q_values[0])

    def train_model(self):
        if len(self.memory.buffer) < self.train_start:
            return

        # 更新目标网络参数
        self.step += 1
        if self.step % self.update_fre == 0:
            self.target_model.set_weights(self.model.get_weights())

        # 从记忆库中取出一个batch
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample()
        # 计算TD目标值
        target = self.model.predict(state_batch, batch_size=self.memory.batch_size)
        target_val = self.target_model.predict(next_state_batch, batch_size=self.memory.batch_size)

        for i in range(self.memory.batch_size):
            action, reward, done = action_batch[i], reward_batch[i], done_batch[i]
            if not done:
                target[i][action] = reward + self.gamma * np.max(target_val[i])
            else:
                target[i][action] = reward

        # 训练模型
        self.model.fit(state_batch, target, batch_size=self.memory.batch_size, epochs=1, verbose=0, callbacks=[self.history])
        self.losses_lsit.append(self.history.losses[0])


class Memory:

    def __init__(self, memory_size, batch_size):
        self.buffer = deque(maxlen=memory_size)
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, done):
        # 如果队列满，再从队尾添加数据则可以自动从队首踢出数据
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        bach_size = min(self.batch_size, len(self.buffer))
        minibach = random.sample(self.buffer, bach_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.stack, zip(*minibach))
        # 调整输出格式 （bach_size，1,4）-> (bach_size,4)

        state_batch = np.reshape(state_batch, [state_batch.shape[0], state_batch.shape[2]])
        next_state_batch = np.reshape(next_state_batch, [next_state_batch.shape[0], next_state_batch.shape[2]])
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


# 绘制得分图
def draw_score_plot(scores, foldername='./graph', filename='graph.png'):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('Score')
    ax1.set_xlabel('episode')
    ax1.set_ylabel('score')
    ax1.plot(range(len(scores)), scores, color='blue')
    plt.savefig(os.path.join(foldername, filename))


# 绘制损失函数
def draw_loss_plot(losses, foldername='./graph', filename='graph.png'):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('Loss')
    ax1.set_xlabel('episode')
    ax1.set_ylabel('loss')
    ax1.plot(range(len(losses)), losses, color='blue')
    plt.savefig(os.path.join(foldername, filename))
