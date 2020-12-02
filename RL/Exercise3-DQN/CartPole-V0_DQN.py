import gym
from DQN import *

MODEL_NAME = "./model"
GRAPH_NAME = "./graph"
DATA_NAME = "./data"


def train(max_iter=200, max_episodes=1000):
    env = gym.make('CartPole-v0')
    # 使用unwrapped可以得到原始的类
    env = env.unwrapped
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = Agent(state_size, action_size, batch_size=64, train_start=100, render=False)

    # 检查当前目录下是否存在 model graph data 文件夹
    if not os.path.exists('model'):
        os.makedirs(os.path.join(os.getcwd(), 'model'))
        print('make model folder')
    if not os.path.exists('graph'):
        os.makedirs(os.path.join(os.getcwd(), 'graph'))
    if not os.path.exists('data'):
        os.makedirs(os.path.join(os.getcwd(), 'data'))

    # 如果存在模型文件，则加载
    model_path = os.path.join(MODEL_NAME, 'cartpole_dqn.h5')
    target_model_path = os.path.join(MODEL_NAME, 'cartpole_tdqn.h5')
    if os.path.exists(model_path):
        print('model is loaded')
        agent.model.load_weights(model_path)
    if os.path.exists(target_model_path):
        print('target model is loaded')
        agent.target_model.load_weights(target_model_path)

    scores, losses = [], []

    for epoch in range(max_episodes):
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        if np.mean(scores[-5:]) > 1000:
            break
        for it in range(max_iter):
            if agent.render:
                env.render()

            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            score += reward

            if done:
                reward = -100

            agent.memory.push(state, action, reward, next_state, done)
            agent.train_model()
            state = next_state

            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            if done or score >= max_iter:
                break

        if len(agent.losses_lsit) > 0:
            loss = np.array(agent.losses_lsit)
            print("(episode: {}; score: {}; memory length: {}; loss-mean: {})"
                  .format(epoch, score, len(agent.memory.buffer), loss.mean()))
            scores.append(score)
            losses.append(loss.mean())

        if epoch % 50 == 0:
            agent.model.save_weights(model_path)
            agent.target_model.save_weights(target_model_path)

    scores_path = os.path.join(DATA_NAME, 'scores.npy')
    losses_path = os.path.join(DATA_NAME, 'losses.npy')
    np.save(scores_path, np.array(scores))
    np.save(losses_path, np.array(losses))
    agent.model.save_weights(model_path)
    agent.target_model.save_weights(target_model_path)
    draw_score_plot(scores, foldername=GRAPH_NAME, filename='train_scores.png')
    draw_loss_plot(losses, foldername=GRAPH_NAME, filename='losses.png')


def test(max_iter=2000, max_episodes=100):
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = Agent(state_size, action_size, batch_size=64, epsilon=0.000, render=False)

    model_path = os.path.join(MODEL_NAME, 'cartpole_dqn.h5')
    if os.path.exists(model_path):
        print('loading is ok')
        agent.model.load_weights(model_path)

    scores = []

    for epoch in range(max_episodes):
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        while score <= max_iter:
            if agent.render:
                env.render()
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            score += reward
            # 在CartPole_v0这个env中reward只有0/1两种,0只存在与done(游戏结束的情况)
            state = next_state

            if done or score >= max_iter:
                print("(episode: {}; score: {};)"
                      .format(epoch, score))
                scores.append(score)
                break
    test_scores_path = os.path.join(DATA_NAME, 'test_scores.npy')
    np.save(test_scores_path, np.array(scores))
    draw_score_plot(scores, foldername=GRAPH_NAME, filename='test_scores.png')


if __name__ == '__main__':
    train(max_iter=200, max_episodes=1000)
    # test(max_iter=200, max_episodes=100)