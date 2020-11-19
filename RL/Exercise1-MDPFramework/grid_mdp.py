import logging
import numpy
import random
from gym import spaces
import gym

logger = logging.getLogger(__name__)

class GridEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        # State space
        self.states = [1, 2, 3, 4, 5, 6, 7, 8]

        # Termination state
        self.terminate_states = dict()
        self.terminate_states[6] = 1
        self.terminate_states[7] = 1
        self.terminate_states[8] = 1

        # Action space
        self.actions = ['n', 'e', 's', 'w']

        # Reward
        self.rewards = dict()
        self.rewards['1_s'] = -1.0
        self.rewards['3_s'] = 1.0
        self.rewards['5_s'] = -1.0

        # State transform
        self.t = dict()
        self.t['1_s'] = 6
        self.t['1_e'] = 2
        self.t['2_w'] = 1
        self.t['2_e'] = 3
        self.t['3_s'] = 7
        self.t['3_w'] = 2
        self.t['3_e'] = 4
        self.t['4_w'] = 3
        self.t['4_e'] = 5
        self.t['5_s'] = 8
        self.t['5_w'] = 4

        self.gamma = 0.8
        self.viewer = None
        self.state = None

        self.x = [140, 220, 300, 380, 460, 140, 300, 460]
        self.y = [250, 250, 250, 250, 250, 150, 150, 150]

    def getTerminal(self):
        return self.terminate_states

    def getGamma(self):
        return self.gamma

    def getStates(self):
        return self.states

    def getAction(self):
        return self.actions

    def getTerminate_states(self):
        return self.terminate_states

    def setAction(self, s):
        self.state = s

    def reset(self):
        self.state = self.states[int(random.random() * len(self.states))]
        return self.state

    def step(self, action):
        # Current state
        state = self.state
        # Judge system is whether terminated
        if state in self.terminate_states:
            return state, 0, True, {}
        key = "%d_%s" % (state, action)
        # Determinate the next state
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state
        # update state
        self.state = next_state
        is_terminal = False
        # Judge system is whether terminated in the next state
        if next_state in self.terminate_states:
            is_terminal = True
        # Determinate the reward
        if key not in self.rewards:
            r = 0.0
        else:
            r = self.rewards[key]
        return next_state, r, is_terminal, {}

    def transform(self, state, action):
        # Judge system is whether terminated
        if state in self.terminate_states:
            return state, 0, True, {}
        key = "%d_%s" % (state, action)
        # Determinate the next state
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state
        is_terminal = False
        # Judge system is whether terminated in the next state
        if next_state in self.terminate_states:
            is_terminal = True
        # Determinate the reward
        if key not in self.rewards:
            r = 0.0
        else:
            r = self.rewards[key]
        return next_state, r, is_terminal, {}

    def render(self, mode='human', close=False):
        # If close is True, close the created viewer
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
                return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            screen_width, screen_height = (600, 400)
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # Create grid world
            self.line1 = rendering.Line((100, 300), (500, 300))
            self.line2 = rendering.Line((100, 200), (500, 200))
            self.line3 = rendering.Line((100, 300), (100, 100))
            self.line4 = rendering.Line((180, 300), (180, 100))
            self.line5 = rendering.Line((260, 300), (260, 100))
            self.line6 = rendering.Line((340, 300), (340, 100))
            self.line7 = rendering.Line((420, 300), (420, 100))
            self.line8 = rendering.Line((500, 300), (500, 100))
            self.line9 = rendering.Line((100, 100), (180, 100))
            self.line10 = rendering.Line((260, 100), (340, 100))
            self.line11 = rendering.Line((420, 100), (500, 100))
            # Create the first skeleton
            self.kulo1 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(140, 150))
            self.kulo1.add_attr(self.circletrans)
            self.kulo1.set_color(0, 0, 0)
            # Create the second skeleton
            self.kulo2 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(460, 150))
            self.kulo2.add_attr(self.circletrans)
            self.kulo2.set_color(0, 0, 0)
            # Create the bullion
            self.gold = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(300, 150))
            self.gold.add_attr(self.circletrans)
            self.gold.set_color(1, 0.9, 0)
            # Create the robot
            self.robot = rendering.make_circle(30)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0.8, 0.6, 0.4)

            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)
            self.line5.set_color(0, 0, 0)
            self.line6.set_color(0, 0, 0)
            self.line7.set_color(0, 0, 0)
            self.line8.set_color(0, 0, 0)
            self.line9.set_color(0, 0, 0)
            self.line10.set_color(0, 0, 0)
            self.line11.set_color(0, 0, 0)

            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            self.viewer.add_geom(self.line11)
            self.viewer.add_geom(self.kulo1)
            self.viewer.add_geom(self.kulo2)
            self.viewer.add_geom(self.gold)
            self.viewer.add_geom(self.robot)

        if self.state is None:
            return None
        self.robotrans.set_translation(self.x[self.state-1], self.y[self.state-1])

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))
