"""
强化学习q learning flappy bird
"""
from game.wrapped_flappy_bird import GameState
import time
import numpy as np 
import skimage.color
import skimage.transform
import skimage.exposure
import tensorflow as tf 
import random 
import argparse

# 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--model-only", help="加载已有模型，不随机探索，仍旧训练", action='store_true')
args = parser.parse_args()

# 测试用代码
def _test_save_img(img):
    # 把每一帧图片存储到文件里，调试用
    from PIL import Image
    im = Image.fromarray((img*255).astype(np.uint8), mode='L') # 图片已经被处理为0~1之间的亮度值，所以*255取整数变灰度展示
    im.save('./img.jpg')

# 构建卷积神经网络
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

    # 尝试加载之前保存的模型参数
    try:
        model.load_weights('./weights.h5')
        print('加载模型成功...................')
    except:
        pass
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
    stat_t = np.stack([img_t] * 4, axis=2)
    return stat_t 

# 初始状态
stat_t = reset_stat()
# 训练样本
transitions = []

# 时刻
t = 0

# 随机探索的概率控制
INIT_EPSILON = 0.1
FINAL_EPSILON = 0.005
EPSLION_DELTA = 1e-6
# 最大留存样本个数
TRANS_CAP =  20000
# 至少有多少样本才训练
TRANS_SIZE_FIT = 10000
# 训练集大小
BATCH_SIZE = 32
# 未来激励折扣
GAMMA = 0.99

# 随机探索概率
if args.model_only: # 不随机探索（极低概率）
    epsilon = FINAL_EPSILON
else:
    epsilon = INIT_EPSILON

# 打印一些进度信息
rand_flap =0    # 随机点击次数
rand_noflap = 0 # 随机不点击次数
model_flap=0    # 模型点击次数
model_noflap=0  # 模型不点击次数
model_train_times = 0   # 模型训练次数

# 游戏启动
while True:    
    # 动作
    action_t = [0,0]

    action_type = '随机'

    # 随着学习，降低随机探索的概率，让模型趋于稳定
    if (t <= TRANS_SIZE_FIT and not args.model_only) or random.random() <= epsilon:
        n = random.random()
        if n <= 0.95:
            action_index = 0
            rand_noflap+=1
        else:
            action_index = 1
            rand_flap+=1
        #print('[随机探索] t时刻进行随机动作探索...')
    else: # 模型预测2个操作的未来累计回报
        action_type = '经验'
        Q_t = model.predict(np.expand_dims(stat_t, axis=0))[0]
        action_index = np.argmax(Q_t)   # 回报最大的action下标
        if action_index==0:
            model_noflap+=1
        else:
            model_flap+=1
        #print('[已有经验] 预测t时刻2个动作的未来总回报 -- 不点击:{} 点击:{}'.format(Q_t[0], Q_t[1]))

    action_t[action_index] = 1
    #print('时刻t将执行的动作为{}'.format(action_t))

    # 执行当前动作，返回操作后的图片、本次激励、游戏是否结束
    img_t1, reward, terminal = run_one_frame(action_t)
    _test_save_img(img_t1)
    img_t1 = img_t1.reshape((80,80,1)) # 增加通道维度，因为我们要最近4帧作为4通道图片，用作卷积模型输入
    stat_t1 = np.append(stat_t[:,:,1:], img_t1, axis=2) # 80*80*4，淘汰当前的第0通道，添加最新t1时刻到第3通道

    # 收集训练样本（保留有限的）
    transitions.append({
        'stat_t': stat_t,   # t时刻状态
        'stat_t1': stat_t1, # t1时刻状态
        'reward': reward,   # 本次动作的激励得分
        'terminal': terminal,   # 执行动作后游戏是否结束（ps: 结束意味着没有未来激励了）
        'action_index': action_index,   # 执行了什么动作（0:不点击，1:点击）
    })
    if len(transitions) > TRANS_CAP:
        transitions.pop(0)
    
    # 游戏结束则重置stat_t
    if terminal:
        stat_t = reset_stat()
        #print('死了!!!!!!! 状态t重置为初始帧...')
    else:   # 否则切为新的状态
        stat_t = stat_t1
        #print('没死~~~ 状态t切换为状态t1...')

    # 过了观察期，开始训练
    if t >= TRANS_SIZE_FIT and t % 10 == 0:
        minibatch = random.sample(transitions, BATCH_SIZE)
        # 模型训练的输入：t时刻的状态(最近4帧图片)
        inputs_t = np.concatenate([tran['stat_t'].reshape((1,80,80,4)) for tran in minibatch])
        #print('inputs_t shape', inputs_t.shape)
        ######################################################
        # 模型训练的输出：t时刻的未来总激励（Q_t = reward+gamma*Q_t1）
        # 1，让模型预测t时刻2种action的未来总激励
        Q_t = model.predict(inputs_t, batch_size=len(minibatch))
        # 2，让模型预测t1时刻2种action的未来总激励
        input_t1 = np.concatenate([tran['stat_t1'].reshape((1,80,80,4)) for tran in minibatch])
        Q_t1 = model.predict(input_t1, batch_size=len(minibatch))
        # 3，保留t1时刻2个action中最大的未来总激励
        Q_t1_max = [max(q) for q in Q_t1]
        # 4，t时刻进行action_index动作得到真实激励
        reward_t = [tran['reward'] for tran in minibatch]
        # 5，t时刻进行了什么action
        action_index_t = [tran['action_index'] for tran in minibatch]
        # 6，t1时刻是否死掉了
        terminal = [tran['terminal'] for tran in minibatch]
        # 7，修正训练的目标Q_t=reward+gamma*Q_t1
        # （t时刻action_index的未来总激励=action_index真实激励+t1时刻预测的最大未来总激励）
        for i in range(len(minibatch)):
            if terminal[i]:
                Q_t[i][action_index_t[i]] = reward_t[i] # 因为t1时刻已经死了，所以没有t1之后的累计激励
            else:
                Q_t[i][action_index_t[i]] = reward_t[i] + GAMMA*Q_t1_max[i] # Q_t=reward+Q_t1
        # print('Q_t shape', Q_t.shape)
        # 训练一波
        #print(inputs_t)
        #print(Q_t)
        model.fit(inputs_t, Q_t, batch_size=len(minibatch))
        model_train_times += 1
        # 训练1次则降低些许的随机探索概率
        if epsilon > FINAL_EPSILON:
            epsilon -= EPSLION_DELTA
        
        # 每5000次batch保存一次模型权重（不适用saved_model，后续加载只会加载权重，模型结构还是程序构造，因为这样可以保持keras model的api)
        if model_train_times % 5000 == 0:
            model.save_weights('./weights.h5')

        ######################################################
    if t % 100 == 0:
        print('总帧数:{} 剩余探索概率:{}% 累计训练次数:{} 累计随机点:{} 累计随机不点:{} 累计模型点:{} 累计模型不点:{} 训练集:{} '.format(
            t, round(epsilon * 100, 4), model_train_times, rand_flap, rand_noflap, model_flap, model_noflap,
            len(transitions)))
    t = t + 1
    #time.sleep(1)