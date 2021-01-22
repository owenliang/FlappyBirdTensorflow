"""
强化学习q learning flappy bird
"""
from game.wrapped_flappy_bird import GameState
from random import random
import time
import numpy as np 
import skimage.color
import skimage.transform
import skimage.exposure
import tensorflow as tf 
import random 

def build_model():
    # 卷积神经网络：https://blog.csdn.net/FontThrone/article/details/76652753
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(80,80,4)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(8, 8), padding='same',strides=4, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), padding='same',strides=2, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same',strides=1, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(2), # 对应2个action未来总回报预期
    ])
    model.compile(loss='mse', optimizer='adam')
    return model

# 创建游戏
game = GameState()
# 卷积模型
model = build_model()

# 执行1帧游戏
def run_one_frame(action):
    global game 
    # image_data：执行动作后的图像（288*512*3的RGB三维数组）
    # reward：本次动作的奖励
    # terminal：游戏是否失败
    img, reward, terminal = game.frame_step(action)
    # RGB转灰度图
    img = skimage.color.rgb2gray(img)
    # 压缩到80*80的图片（根据RGB算出来的亮度，其数值很小）
    img = skimage.transform.resize(img, (80,80))
    # 把亮度标准化到0~1之间，用作模型输入
    img = skimage.exposure.rescale_intensity(img, out_range=(0,1))
    return img,reward,terminal

# 强化学习初始化状态
def reset_stat():
    # 执行第一帧，不点击
    img_t,_,_ =  run_one_frame([1,0])
    # 卷积网络的输入是连续的4帧游戏画面，对于首帧只能重复4遍
    stat_t = np.stack([img_t] * 4, axis=2).reshape((1,80,80,4))
    return stat_t 

# 初始状态
stat_t = reset_stat()
# 训练样本
transitions = []

# 游戏启动
while True:
    # 动作
    action_t = np.array([0,0])

    # TODO: 引入随机动作进行探索
    if random.random() <= 0.1:
        action_index = random.randint(0,1)
        print('t时刻进行随机动作探索...')
    else: # 模型预测2个操作的未来累计回报
        Q_t = model.predict(stat_t)[0]
        action_index = np.argmax(Q_t)   # 回报最大的action下标
        print('预测t时刻2个动作的未来总回报 -- 不点击:{} 点击:{}'.format(Q_t[0], Q_t[1]))
    
    action_t[action_index] = 1
    print('时刻t将执行的动作为{}'.format(action_t))

    # 执行当前动作，返回操作后的图片、本次激励、游戏是否结束
    img_t1, reward, terminal = run_one_frame(action_t)
    img_t1 = img_t1.reshape((1,80,80,1))
    stat_t1 = np.append(stat_t[:,:,:,1:], img_t1, axis=3) # 1*80*80*4，淘汰当前的第0通道，添加最新t1时刻到第3通道

    # 收集训练样本（保留有限的）
    transitions.append({
        'stat_t': stat_t,
        'stat_t1': stat_t1,
        'reward': reward, 
        'terminal': terminal,
    })
    if len(transitions) > 50000:
        transitions.pop(0)
    
    # 游戏结束则重置stat_t
    if terminal:
        stat_t = reset_stat()
        print('死了!!!!!!! 状态t重置为初始帧...')
    else:   # 否则切为新的状态
        stat_t = stat_t1
        print('没死~~~ 状态t切换为状态t1...')

    # # 打开下面代码可以观察每帧图片
    # from PIL import Image
    # im = Image.fromarray(img_t1.astype(np.uint8), mode='L')
    # im.save('./img_t1.jpg')

    time.sleep(0.5)
