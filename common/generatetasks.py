# speed单位 m/s，时间最小间隔为秒
import numpy as np
import random
class agent():
    def __init__(self, speed):
        self.speed = speed

class UAV(agent):
    def __init__(self, speed, type):
        super(UAV, self).__init__(speed)
        self.type = type
#任务数量
tasknum=4
#电台数量
radionum = 2
#最大任务步长
stepmax = 200
#最小任务步长
stepmin=150
#world
map = 'b500'
#世界大小
size = [500, 500, 100]
#地面高度可以常设为20
ground = 20
#weidu
POS_D = 3
def generateTasks():
    dict = {}
    tasks=[]
    steps = []
    basepos = np.random.randint(0,size)
    tasks.append(list(basepos))
    maxstep = 0
    for i in range(tasknum):
        pos = []
        x = random.randint(0, size[0])
        y = random.randint(0, size[1])
        pos.append(x)
        pos.append(y)
        step = np.random.randint(stepmin, stepmax)
        steps.append(step)
        maxstep = max(step, maxstep)
        for j in range(step):
            action = np.random.randint(0,9)
            pos.append(action)
        tasks.append(pos)
    dict["step"] = steps
    dict["maxstep"] = maxstep
    dict["tasknum"] = tasknum
    dict["radionum"] = radionum
    dict["map"] = map
    dict["task"] = tasks
    return dict

def generate3Dtasks():
    dict = {}
    tasks=[]
    steps = []
    basepos = np.random.randint(0,size)

    return dict

a = np.random.randint(3)
a = a / np.linalg.norm(a)

if __name__ == '__main__':
    a = generateTasks()
    print(a)

