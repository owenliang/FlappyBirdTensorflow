import numpy as np
import sys
import random
import pygame
from . import flappy_bird_utils
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle

# 屏幕宽*高
FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512

# 初始化游戏
pygame.init()
FPSCLOCK = pygame.time.Clock()  # FPS限速器
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))   # 宽*高
pygame.display.set_caption('Flappy Bird')   # 标题

# 加载素材
IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()

PIPEGAPSIZE = 100 # 上下水管之间的距离是固定的100像素
BASEY = SCREENHEIGHT * 0.79 # 地面图片的y坐标

# 小鸟图片的宽*高
PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
# 水管图片的宽*高
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()

# 背景图片的宽
BACKGROUND_WIDTH = IMAGES['background'].get_width()

# 小鸟图片动画播放顺序
PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

# Flappy bird游戏类
class GameState:
    def __init__(self):
        self.score = 0
        self.playerIndex = 0
        self.loopIter = 0

        # 玩家初始坐标
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)

        # 地面图片需要跑马灯效果，它比屏幕宽一点，每帧向左移动，当要耗尽时重新回到右边，如此往复
        self.basex = 0         # 地面图片的x坐标
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH  # 地面图片比屏幕宽度长多少像素，就是它可以移动的距离

        newPipe1 = getRandomPipe()  # 生成一对上下管子
        newPipe2 = getRandomPipe()  # 再生成一对上下管子

        # 上面2根管子，都放到屏幕右侧之外，x相邻半个屏幕距离
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        # 下面2根管子，都放到屏幕右侧之外，x相邻半个屏幕距离
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        # 水管的水平移动速度，每次x-4实现向左移动
        self.pipeVelX = -4

        # 小鸟Y方向速度
        self.playerVelY    =  0
        # 小鸟Y方向重力加速度，每帧作用域playerVelY，令其Y速度向下加大
        self.playerAccY    =   1
        # 点击后，小鸟Y方向速度重置为-9，也就是开始向上移动
        self.playerFlapAcc =  -9

        # 小鸟Y方向速度限制
        self.playerMaxVelY =  10   # Y向下最大速度10

    # 执行一次操作，返回操作后的画面、本次操作的奖励（活着+0.1，死了-1，飞过水管+1）、游戏是否结束
    def frame_step(self, input_actions):
        # 给pygame对积累的事件做一下默认处理
        pygame.event.pump()

        # 活着就奖励0.1分
        reward = 0.01
        # 是否死了
        terminal = False

        # 必须传有效的action，[1,0]表示不点击，[0,1]表示点击，全传0是不对的
        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # 每3帧换一次小鸟造型图片，loopIter统计经过了多少帧
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter += 1

        # 让地面向左移动，游戏开始的时候地面x=0，逐步减小x
        if self.basex + self.pipeVelX <= -self.baseShift:
            self.basex = 0
        else: # 图片即将滚动耗尽，重置x坐标
            self.basex += self.pipeVelX

        # 点击了屏幕
        if input_actions[1] == 1:
            self.playerVelY = self.playerFlapAcc # 将小鸟y方向速度重置为-9，也就是向上移动
            #SOUNDS['wing'].play()   # 播放扇翅膀的声音
        elif self.playerVelY < self.playerMaxVelY:  # 没点击屏幕并且没达到最大掉落速度，继续施加重力加速度
            self.playerVelY += self.playerAccY

        # 将速度施加到小鸟的y坐标上
        self.playery += self.playerVelY
        if self.playery < 0:    # 撞到上边缘不算死
            self.playery = 0 # 限制它别飞出去
        elif self.playery + PLAYER_HEIGHT >= BASEY: # 小鸟碰到地面
            self.playery = BASEY - PLAYER_HEIGHT # 限制它别穿地

        # 让上下水管都向左移动一次
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # 判断小鸟是否穿过了一排水管，因为上下水管x一样，只需要用上排水管判断
        playerMidPos = self.playerx + PLAYER_WIDTH / 2  # 小鸟中心的x坐标（这个是固定值，小鸟实际不会动，是水管在动）
        for pipe in self.upperPipes:    # 检查与上排水管的关系
            pipeMidPos = pipe['x'] + PIPE_WIDTH / 2 # 水管中心的x坐标
            if pipeMidPos <= playerMidPos < pipeMidPos + abs(self.pipeVelX): # 小鸟x坐标刚刚飞过了水管x中心（4是水管的移动速度）
                self.score += 1 # 游戏得分+1
                #SOUNDS['point'].play()
                reward = 100  # 产生强化学习的动作奖励10分

        # 最左侧水管马上离开屏幕，生成新水管
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # 最左侧水管彻底离开屏幕，删除它的上下2根水管
        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # 检查小鸟是否碰到水管
        isCrash= checkCrash({'x': self.playerx, 'y': self.playery, 'index': self.playerIndex}, self.upperPipes, self.lowerPipes)
        if isCrash:  # 死掉了
            #SOUNDS['hit'].play()
            #SOUNDS['die'].play()
            reward = -10 # 负向激励分
            terminal = True # 本次操作导致游戏结束了

        ##### 进入重绘 #######

        # 贴背景图
        SCREEN.blit(IMAGES['background'], (0,0))
        # 画水管
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))
        # 画地面
        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        # 画得分（训练时候别打开，造成干扰了）
        #showScore(self.score)
        # 画小鸟
        SCREEN.blit(IMAGES['player'][self.playerIndex], (self.playerx, self.playery))
        # 重绘
        pygame.display.update()
        # 留存游戏画面（截图是列优先存储的，需要转行行优先存储）
        # https://stackoverflow.com/questions/34673424/how-to-get-numpy-array-of-rgb-colors-from-pygame-surface
        image_data = pygame.surfarray.array3d(pygame.display.get_surface()).swapaxes(0,1)
        # 死亡则重置游戏状态
        if terminal:
            self.__init__()
        # 控制FPS
        FPSCLOCK.tick(FPS)
        return image_data, reward, terminal

# 生成一对水管，放到屏幕外面
def getRandomPipe():
    gapY = random.randint(70, 140)

    # 注：每一对水管的缝隙高度都是一样的PIPEGAPSIZE，gayY决定的是缝隙的上边缘y坐标
    pipeX = SCREENWIDTH + 10    # 水管出现在屏幕右侧之外

    return [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT},  # 计算上面水管图片的y坐标，就是缝隙上边缘y减去水管本身高度
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # 计算下面水管图片的y坐标，就是缝隙上边缘y加上缝隙本身高度
    ]

# 检查小鸟是否碰到水管或者地面（天花板不算）
def checkCrash(player, upperPipes, lowerPipes):
    pi = player['index']    # 小鸟的第几张图片

    # 图片的宽*高
    player['w'] = IMAGES['player'][pi].get_width()
    player['h'] = IMAGES['player'][pi].get_height()

    # 小鸟碰到了地面
    if player['y'] + player['h'] >= BASEY - 1:
        return True
    else: # 小鸟与水管进行碰撞检测
        # 小鸟图片的矩形区域
        playerRect = pygame.Rect(player['x'], player['y'], player['w'], player['h'])

        # 每一对水管
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # 上面水管的矩形
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            # 下面水管的矩形
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

            # 小鸟图片的非透明像素掩码
            pHitMask = HITMASKS['player'][pi]
            # 上水管的非透明像素掩码
            uHitmask = HITMASKS['pipe'][0]
            # 下水管的非透明像素掩码
            lHitmask = HITMASKS['pipe'][1]

            # 检测小鸟与上面水管的碰撞
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            # 检测小鸟与下面水管的碰撞
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return True
    return False


# 2个矩形区域的碰撞检测
def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    # 求2个矩形相交的矩形区域
    rect = rect1.clip(rect2)

    # 相交面积为0
    if rect.width == 0 or rect.height == 0:
        return False

    # 相交矩形x,y相对于2个矩形左上角的距离
    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    # 检查相交矩形内的每个点，是否在2个矩形内同时是非透明点，那么就碰撞了
    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

# 展示得分，传入一个整数得分
def showScore(score):
    # 转成单个数字的列表
    scoreDigits = [int(x) for x in list(str(score))]

    # 计算展示所有数字要占多少像素宽度
    totalWidth = 0
    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    # 计算绘制起始x坐标
    Xoffset = (SCREENWIDTH - totalWidth) / 2

    # 逐个数字绘制
    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, 20))    # y坐标贴近屏幕上边缘
        Xoffset += IMAGES['numbers'][digit].get_width() # 移动绘制x坐标