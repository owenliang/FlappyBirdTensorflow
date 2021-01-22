"""
演示pygame制作的flappy bird如何逐帧调用执行
"""
from game.wrapped_flappy_bird import GameState
from random import random
import time

# 创建游戏
game = GameState()

# 游戏启动
while True:
    r = random()
    if r <= 0.92:  # 92%的概率不点击屏幕
        game.frame_step([1,0]) # 动作：[1,0] 表示不点击
    else: # 8%的概率点击屏幕
        game.frame_step([0,1]) # 动作：[0,1] 表示点击