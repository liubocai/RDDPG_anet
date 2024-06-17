import os

import gym
from gym import spaces
import numpy as np
from shapely import geometry as geo
import pygame
import cv2
#1.The variables representing the coordinates are represented by nd.array
#  and then converted to geo objects in the geometry calculation
#todo 2.when to terminated?
#目的是保证视频流的传输，所以网速不能过低，这个网速包括电台与感知设备，电台与基地。但是

#目前的map是二维的二值map，以后可能有更真实的三维或者dem等地图，相应的坐标、网速计算方式可能都得改

class GroundEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 6}

    def __init__(self, tasks=[], map='b.npy'):
        self.render_mode = "human"
        #1.Environment

        self.map = np.load('/home/inspur2/workspace/gym-examples/gym_examples/map/b.npy')
        self.size = np.array(self.map.shape) #(rows, columns)
        self.radiospeed = 5
        #2.Tasks
        self.tasks = tasks
        self.basepos = [] #basement's position
        self.cliposs = np.empty(shape=[0,2]) #current client's position
        self.tasknum = 0  #tasks'number
        self.count = 0    #current step
        self.maxstep = 0  #max step in all task
        self.speed_threshold = 4.13
        self.vid = cv2.VideoWriter('/home/inspur2/2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 6, (154,154) )
        self.clients_paths = [[0,0],
                              [500,0,2,2,2,2,2,2,2,2,2,1,3,2,1,4,1,2,3,4,6,7,8,5,7,2,3,6,7,8],
                              [0,500,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

        #3.action and observation
        self.action_space = spaces.Discrete(9)
        # 8 actions, from "right" counterclockwise rotation
        # todo 9项动作空间的设计暂时具有局限性，每步1m的变化太小，并且不是所有的设备运动速度都相同，需要考虑时间和速度
        self._action_to_direction = {
            0: np.array([1, 0]), #向右前进1m
            1: np.array([1, 1]),
            2: np.array([0, 1]),
            3: np.array([-1, 1]),
            4: np.array([-1, 0]),
            5: np.array([-1, -1]),
            6: np.array([0, -1]),
            7: np.array([1, -1]),
            8: np.array([0, 0]), #no action, stop
        }
        # radio's position in this ground env
        self.observation_space = spaces.Box(low=np.array([0,0]), high = np.array([i-1 for i in self.size]), shape=(2,), dtype=int)

        #4.visiable
        self.window = None
        self.window_size_x = self.size[1]
        self.window_size_y = self.size[0] #int(self.window_size_x * self.size[1] / self.size[0])
        self.clock = None

    def _parser_tasks(self):
        self.basepos = np.array(self.tasks[0])
        self.tasknum = len(self.tasks)-1
        self.cliposs = np.empty(shape=[0,2])
        for item in self.tasks[1:]:
            if self.maxstep < len(item)-2:
                self.maxstep = len(item)-2
            self.cliposs = np.append(self.cliposs, [item[:2]], axis=0)





    def _get_obs(self):
        # l = [cli, self._radio_position, self.basepos] for cli in self.cliposs
        s = np.row_stack([self.cliposs, self._radio_position.reshape((1,2)), self.basepos.reshape((1,2))])
        s = s / self.size #normalize
        s = s.flatten()

        # for cli in self.cliposs:
        #     s.append(cli[0] / self.size[0])
        # cli = [c for c in self.cliposs]
        # cli.append(self._radio_position)
        # cli.append(self.basepos)
        return s.copy()

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        self._parser_tasks()
        self._radio_position = self.basepos
        self.count = 0
        observation = self._get_obs()
        if self.render_mode != "none":
            self._render_frame()

        return observation


    def step(self, action):
        #1.update radio's pos
        direction = self._action_to_direction[action] *self.radiospeed
        self._radio_position = np.clip(
            self._radio_position + direction, 0, [i-1 for i in self.size]
        )
        #2.update cli position
        for i in range(self.tasknum):
            if self.count+2 < len(self.tasks[i+1]):
                clidirection = self._action_to_direction[self.tasks[i+1][self.count+2]] * 5
                self.cliposs[i] = np.clip(self.cliposs[i] + clidirection, 0 , self.size-1)
        #3.if is terminated
        self.count += 1
        speeds = []
        for cli in self.cliposs:
            speeds.append(self.getNetFromTif(self._radio_position, cli))
        speeds.append(self.getNetFromTif(self._radio_position, self.basepos))
        speeds = np.array(speeds)
        norm_speeds = speeds / 41.2771 #希望数据大
        std = np.std(norm_speeds)      #希望网速能平均，减小分散
        mean = np.mean(norm_speeds)
        minspeed = min(speeds)
        meanspeed = np.mean(speeds)
        terminated = bool(
            # minspeed < self.speed_threshold
            self.count > self.maxstep
        )
        # ended = bool(
        #     self.count > self.maxstep
        # )
        if not terminated:
            reward = mean / (std + 1)
        else:
            reward = 1.0

        observation = self._get_obs()

        if self.render_mode != "none":
            self._render_frame()
        #4.which infos

        info = []
        info.append(speeds)
        info.append(meanspeed)
        info.append(self._radio_position.copy())



        return observation, reward, terminated, False, info

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size_x, self.window_size_y))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # canvas = pygame.Surface((self.window_size_x, self.window_size_y))
        # canvas.fill((255, 255, 255))
        canvas = pygame.image.load('/home/inspur2/workspace/gym-examples/gym_examples/map/b.jpg')
        pix_square_size = (
            self.window_size_x / self.size[0]
        )  # The size of a single grid square in pixels

        # First we draw the points
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.basepos,
                (10, 10),
            ),
        )
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * self._radio_position,
                (10, 10),
            ),
        )
        for cli in self.cliposs:
            pygame.draw.rect(
                canvas,
                (100,100,100),
                pygame.Rect(
                    pix_square_size * cli,
                    (10, 10),
                ),
            )
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            na =  np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
            img = cv2.cvtColor(na, cv2.COLOR_RGB2BGR)
            if self.vid.isOpened():
                self.vid.write(img)
            else:
                self.vid = cv2.VideoWriter(r'/home/inspur2/2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 6, (1071,941) )
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        self.vid.release()

    def _is_terminated(self):
        #when to terminated: every client's job is done or disconnect with any cli's
        for item in self.clients_paths.values():
            if self.count < len(item)-1:
                return False
        return True

    def getNetFromTif(self, point1, point2):
        line = geo.LineString([point1, point2])
        distance_build = 0
        for i in range(int(point1[0]), int(point2[0])):
            for j in range(int(point1[1]), int(point2[1])):
                pixel = geo.Polygon([(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j), (i, j)])
                if self.map[i][j] == 0:
                    continue
                if pixel.intersects(line):
                    distance_build += pixel.intersection(line).length
        if distance_build >= 50.271:
            return 0
        distance_total = line.length
        distance_tree = distance_total - distance_build
        net_speed = max(41.2771 - 0.0149*distance_tree - 0.8211*distance_build - 0.0003795*pow(distance_tree,2), 0)
        return net_speed





















    """
    之前用tree和building来表示环境时的求网速方法
    """
    # def net_speed(self):
    #     line = geo.LineString([self._radio_position,self._cli_position])
    #     x1 = 0
    #     x2 = 0
    #     if self.buildings.intersects(line):
    #         x1 = self.buildings.intersection(line).length
    #     if self.trees.intersects(line):
    #         x2 = self.trees.intersection(line).length
    #     y = 41.2771 - 0.0149*x1 - 0.8211*x2 - 0.0003795*pow(x1,2)
    #     return y
    #
    # def getNetSpeedIn2points(self, point1, point2):
    #     line = geo.LineString([point1, point2])
    #     x1 = 0
    #     x2 = 0
    #     if self.buildings.intersects(line):
    #         x1 = self.buildings.intersection(line).length
    #     if self.trees.intersects(line):
    #         x2 = self.trees.intersection(line).length
    #     y = 41.2771 - 0.0149*x1 - 0.8211*x2 - 0.0003795*pow(x1,2)
    #     return y







