# FlappyBirdTensorflow

项目介绍：[强化学习Deep Q-Network自动玩flappy bird](https://yuerblog.cc/2021/01/26/%e5%bc%ba%e5%8c%96%e5%ad%a6%e4%b9%a0deep-q-network%e8%87%aa%e5%8a%a8%e7%8e%a9flappy-bird/)

## 项目运行

基于python3.8+tensorflow2.4+pygame实现，最好利用conda初始化一个新的python环境。

安装依赖（最好-i指定走阿里云pip镜像）：

```
pip install -r requirements.txt
```

## 体验效果

weights.h5是我训练好的模型，可以直接体验效果：

```
python q_game.py --model-only
```

## 自己训练

删除weights.h5文件，启动训练：

```
python q_game.py
```

模型训练需要6小时+才能收敛稳定，最好睡觉前运行，睡醒后观察效果，模型会自动保存到weights.h5。