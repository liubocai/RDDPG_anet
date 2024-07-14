#todo  第四版
# 1.把电台数量可以设置成任意 ok
# 2.把tasks从list修改成dict ok
# 3.保留从三维地形计算网速的接口
# 4.增加碰撞检测
# 5.修改网速计算的方式（所有到达base的线路中去出现的网速最大的路径作为该点的网速） ok
# 6.修改reward计算方法， ok
import os
import gym
import torch
from gym import spaces
import numpy as np
from shapely import geometry as geo
import pygame
import cv2
from common.speedNet import speednet


PROJECT_PATH = '/home/inspur2/workspace/gym-examples/gym_examples'
#电台移动速度
RADIO_SPEED = 5
#用户移动速度
CLIENT_SPEED = 5
#电台与设备可进行通信的最低阈值
SPEED_THRESHOLD = 4.13
#位置坐标维度(二维世界为x,y, 三维世界为x,y,z)
POS_D = 2
class GroundEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 6}

    def __init__(self, tasks={}, map='b', render_mode='none', label='', actiontype='serial', statetype='image3c'):
        #1.Environment
        self.render_mode = render_mode if render_mode in self.metadata["render_modes"] else 'none'
        self.map = np.load(map if os.path.exists(map) else PROJECT_PATH+'/map/'+map +'.npy')
        self.mapjpg = PROJECT_PATH+'/map/' + map + '.jpg'
        if POS_D == 3:
            maxheight = np.max(self.map[:,:,1])
            self.size = np.append(np.array(self.map.shape[:2]), maxheight) #(rows, columns / maxheight)
        else:
            self.size = np.array(self.map.shape[:2])
        self.radiospeed = RADIO_SPEED
        #2.Tasks
        self.tasks = tasks['task']
        self.radionum = tasks['radionum']
        self.basepos = np.array(tasks['task'][0]) #basement's position
        self.tasknum = len(tasks['step'])  #tasks'number
        self.maxstep = max(tasks['step'])  #max step in all task
        self.cliposs = np.zeros(shape=(self.tasknum, POS_D)) #current client's position
        self.count = 0    #current step
        self.speed_threshold = SPEED_THRESHOLD
        # self.vid = cv2.VideoWriter(PROJECT_PATH + '/video/' + label + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 6, tuple(self.size) )
        self.currentspd = 0
        #3.which action'space
        self.actionType = actiontype
        #net speed fit model
        modeldict = torch.load(
            '/home/inspur2/workspace/datasets/radioSpeedExperiment/radiospeednet_sort_48x3_norm_net2421.pt')
        self.speedmodel = speednet()
        self.speedmodel.net.load_state_dict(modeldict)
        match self.actionType:
            case "serial":
                self.action_space = spaces.Box(low=np.array([0,0]), high=np.array([self.radiospeed, self.radiospeed]), dtype=float) # 连续动作空间，二维的位移，上限是单位时间内的速度 #todo（考虑多个电台的情况，当有多个电台时，动作空间得乘电台数
            case "direction9":
                self.action_space = spaces.Discrete(9)
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
        cr = []
        for item in self.tasks[1:]:
            self.cliposs = np.append(self.cliposs, [item[:2]], axis=0)
            # cr.append(self.getNetFromTif(self.basepos, item[:2]))
        # self.currentspd = sum(cr)





    def _get_obs(self):
        #mask0:no mask, vector as a state
        # s = np.row_stack([self.cliposs, self._radio_position.reshape((1,2)), self.basepos.reshape((1,2))])
        # s = s / self.size #normalize
        # s = s.flatten()

        #mask1:just one pixel to indecate entity, but get a posmask to indicate position
        # posmask = np.zeros(shape=self.size, dtype=np.uint8)
        # posmask[int(self.basepos[0])][int(self.basepos[1])] = 150
        # posmask[int(self._radio_position[0])][int(self._radio_position[1])] = 150
        # for cli in self.cliposs:
        #     posmask[int(cli[0])][int(cli[1])] = 150
        # s = np.stack((posmask, posmask, self.map))

        #mask2:just one pixel to indicate entity, and in the map
        posmask = np.zeros(shape=self.size[:2])
        # posmask = self.map.copy()
        posmask[int(self.basepos[0])][int(self.basepos[1])] = 63
        for radio in self.radioposs:
            posmask[int(radio[0])][int(radio[1])] = 127
        for cli in self.cliposs:
            posmask[int(cli[0])][int(cli[1])] = 191
        if POS_D == 2:
            s = np.stack((posmask, posmask, posmask))
        if POS_D == 3:
            s = np.stack((self.map, posmask), axis=2)
        return s.copy()

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        cr = []

        for i in range(self.tasknum):
            tcpos = np.array(self.tasks[i+1][:2])
            self.cliposs[i] = tcpos
            # cr.append(self.getNetFromTif(self.basepos, tcpos))

        # self.currentspd = sum(cr)
        self.radioposs = np.repeat(self.basepos.reshape(1,2), self.radionum, axis=0)
        self.count = 0
        observation = self._get_obs()
        if self.render_mode != "none":
            self._render_frame()

        return observation


    def step(self, action):
        #0.action
        #todo action应该采取什么数据结构，如果是,如果是action有多种情况，直接的位移，动作是二维三维？或者从网络输出后增加一个数据的封装[[radio1_action],[radio_action]...]
        action = np.reshape(action, (self.radionum, POS_D))
        #1.update radio's pos
        match self.actionType:
            case 'serial':
                for i in range(self.radionum):
                    self.radioposs[i] = np.clip(self.radioposs[i] + action[i], 0, self.size - 1) #todo 需要添加建筑物碰撞机制

            case 'direction9':
                for i in range(self.radionum):
                    direction = self._action_to_direction[action[i][0]] * self.radiospeed
                    self.radioposs[i] = np.clip(self.radioposs[i] + direction, 0, self.size - 1)

        #2.update cli position
        for i in range(self.tasknum):
            if self.count+2 < len(self.tasks[i+1]):
                if POS_D == 2:
                    clidirection = self._action_to_direction[self.tasks[i+1][self.count+2]] * CLIENT_SPEED
                if POS_D == 3:
                    clidirection = np.append(self._action_to_direction[self.tasks[i+1][self.count+2]], 0) * CLIENT_SPEED
                self.cliposs[i] = np.clip(self.cliposs[i] + clidirection, 0 , self.size-1)
        #3.计算网速（全连接的自组网，只求client假想的传数据到基地base的网速，并且最多只经过一跳中继电台，经过中继电台到基地的网速取线段上最低的速度
        radio2base = []
        cli2base = []
        cli2baseidx = []
        for radio in self.radioposs:
            radio2base.append(self.getNetSpeed2d(radio, self.basepos))
        for cli in self.cliposs:
            cr = []
            cr.append(self.getNetSpeed2d(cli, self.basepos))
            for i in range(self.radionum):
                cr.append(min( self.getNetSpeed2d(self.radioposs[i], cli), radio2base[i]))
            cli2base.append(max(cr))
            cli2baseidx.append(cr.index(max(cr)))
        #4.奖励怎么给
        # if min(cli2base) <= self.speed_threshold:
        #     reward = -1
        # elif sum(cli2base) > self.currentspd:
        #     reward = 1
        #     self.currentspd = sum(cli2base)
        # else:
        #     reward = 0
        cli2base = np.array(cli2base)
        cli2base = np.clip(cli2base, 0, 1)
        reward = np.mean(cli2base)
        #3.if is terminated
        self.count += 1
        terminated = bool(
            self.count > self.maxstep
        )


        observation = self._get_obs()

        if self.render_mode != "none":
            self._render_frame()

        #4.which infos
        info = []
        info.append(cli2base)
        info.append(cli2baseidx)
        info.append(self.radioposs.copy())



        return observation, reward, terminated, False, info


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
        net_speed2 = self.speedmodel(torch.FloatTensor([distance_tree, distance_build]))[0]
        return net_speed2

    def genPosMaskBy33(self):
        posmask = np.zeros(shape=self.size[:2], dtype=np.uint8)
        # mask1:with 3x3 box shape to indicate entity
        #1.base
        for i in range(int(self.basepos[0])-1, int(self.basepos[0])+2):
            for j in range(int(self.basepos[1])-1, int(self.basepos[1])+2):
                posmask[i][j] = 150
        #2.radio
        for i in range(int(self._radio_position[1])-1, int(self._radio_position[1])+2):
            posmask[self._radio_position[0]][i] = 50
        for i in range(int(self._radio_position[0]) - 1, int(self._radio_position[0]) + 2):
            posmask[i][self._radio_position[1]] = 50
        #3.clients
        for cli in self.cliposs:
            posmask[int(cli[0])][int(cli[0])] = 100
            posmask[int(cli[0])-1][int(cli[0])-1] = 100
            posmask[int(cli[0])-1][int(cli[0])+1] = 100
            posmask[int(cli[0])+1][int(cli[0])-1] = 100
            posmask[int(cli[0])+1][int(cli[0])+1] = 100
        return posmask

    def genPosMaskByOnePixel(self):
        posmask = np.zeros(shape=self.size, dtype=np.uint8)
        posmask[int(self.basepos[0])][int(self.basepos[1])] = 150
        posmask[int(self._radio_position[0])][int(self._radio_position[1])] = 150
        for cli in self.cliposs:
            posmask[int(cli[0])][int(cli[1])] = 150
        return posmask



    def getNetSpeedFromModel(self, point1, point2):
        return 0

    def getNetSpeed3d(self, point1, point2) -> float:
        # 1.直线栅格化
        grids, dis = self.getGridInLine(point1, point2)
        total = grids.shape[0]
        if total <= 1:
            return 1
        # 2.判断高度以及坐标是否经过建筑并记录个数
        count = 0
        for i in range(total):
            if self.map[grids[i][0]][grids[i][1]][0] != 0 and self.map[grids[i][0]][grids[i][1]][1] >= grids[i][2]:
                count += 1
        # 3.以直线总长除以栅格数量表示每个栅格长度
        d_build = count / total * dis
        d_tree = dis - d_build
        return self.speedmodel(torch.FloatTensor([d_tree, d_build]))[0]

    def getNetSpeed2d(self, point1, point2) -> float:
        # 1.直线栅格化
        grids, dis = self.getGridInLine(point1, point2)
        total = grids.shape[0]
        if total <= 1:
            return 1
        # 2.判断高度以及坐标是否经过建筑并记录个数
        count = 0
        for i in range(total):
            if self.map[grids[i][0]][grids[i][1]] != 0:
                count += 1
        # 3.以直线总长除以栅格数量表示每个栅格长度
        d_build = count / total * dis
        d_tree = dis - d_build
        return self.speedmodel(torch.FloatTensor([d_tree, d_build]))[0]

    def getGridInLine(self, point1, point2):
        v = point2 - point1
        grids = np.empty(shape=(0, point1.shape[0]), dtype ='uint8')
        dis = np.linalg.norm(v, ord=2)
        step = 1.0
        curlength = 0
        while curlength < dis:
            curlength += step
            tp = np.clip(np.fix(point1 + v * curlength / dis + 0.5).reshape(1, point1.shape[0]), 0, self.size[:2]-1).astype(np.int32)
            grids = np.append(grids, tp, axis=0)
        grids = np.unique(grids, axis=0)
        return grids, dis







    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size_x, self.window_size_y))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # canvas = pygame.Surface((self.window_size_x, self.window_size_y))
        # canvas.fill((255, 255, 255))
        canvas = pygame.image.load(self.mapjpg)
        pix_square_size = (
            self.window_size_x / self.size[0]
        )  # The size of a single grid square in pixels

        # First we draw the points
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.basepos,
                (5, 5),
            ),
        )
        for rid in self.radioposs:
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    pix_square_size * rid,
                    (5, 5),
                ),
            )
        for cli in self.cliposs:
            pygame.draw.rect(
                canvas,
                (100,100,100),
                pygame.Rect(
                    pix_square_size * cli,
                    (5, 5),
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
        elif self.render_mode == "rgb_array":  # rgb_array
            na =  np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
            img = cv2.cvtColor(na, cv2.COLOR_RGB2BGR)
            if self.vid.isOpened():
                self.vid.write(img)
            else:
                self.vid = cv2.VideoWriter(r'/home/inspur2/cnn.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 6, self.size)
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        self.vid.release()









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







