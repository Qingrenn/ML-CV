from PG import *
import gym

MODEL_NAME = "./model"
DATA_NAME = './data'


def train(max_iter=200, max_episodes=1000):
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PG(state_size, action_size, render=False, load_model=False)

    if not os.path.exists('model'):
        os.makedirs(os.path.join(os.getcwd(), 'model'))
        print('make model folder')
    if not os.path.exists('data'):
        os.makedirs(os.path.join(os.getcwd(), 'data'))

    # 如果模型存在，则加载模型
    model_path = os.path.join(MODEL_NAME, 'cartpole_pg.h5')
    if os.path.exists(model_path) and agent.load_model:
        print('model is loaded')
        agent.model.load_weights(model_path)

    # 数据保存路径
    scores_path = os.path.join(DATA_NAME, 'scores.npy')
    losses_path = os.path.join(DATA_NAME, 'losses.npy')

    scores, losses = [], []

    for epoch in range(max_episodes):
        score = 0
        state = env.reset()
        for it in range(max_iter):
            # 是否渲染
            if agent.render:
                env.render()
            action, prob = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            # 计算每个状态的回报
            score += reward
            if done:
                reward = -100
            # 保存每个状态的数据
            agent.memorize(state, action, reward, prob)
            # 更新状态
            state = next_state
            # 退出条件
            if done or score >= max_iter:
                agent.train()
                break

        if len(agent.losses_list) > 0:
            loss = np.array(agent.losses_list)
            print("episode:{};score:{};loss-mean:{}".format(epoch, score, 100**2 * loss.mean()))
            scores.append(score)
            losses.append(loss.mean())

        if epoch % 50 == 0:
            agent.model.save_weights(model_path)
            np.save(scores_path, np.array(scores))
            np.save(losses_path, np.array(losses))

    agent.model.save_weights(model_path)
    np.save(scores_path, np.array(scores))
    np.save(losses_path, np.array(losses))
    print('TRAINING END')


def test(max_iter=200, max_episodes=100):
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PG(state_size, action_size, render=False, load_model=True)

    model_path = os.path.join(MODEL_NAME, 'cartpole_pg.h5')
    if os.path.exists(model_path) and agent.load_model:
        print('model is loaded')
        agent.model.load_weights(model_path)

    scores_path = os.path.join(DATA_NAME, 'scores_tst.npy')
    scores = []

    for epoch in range(max_episodes):
        score = 0
        state = env.reset()
        for it in range(max_iter):
            if agent.render:
                env.render()
            action = agent.greedy_action(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            state = next_state
            if done or score >= max_iter:
                break

        print("episode:{};score:{};".format(epoch, score))
        scores.append(score)
        if epoch % 50 == 0:
            np.save(scores_path, np.array(scores))
    np.save(scores_path, np.array(scores))
    print('END TEST')


if __name__ == '__main__':
    train(200, 3000)
    # test(200, 100)