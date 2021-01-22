"""
游戏素材加载
"""
import pygame
import sys
import os

assets_dir = os.path.dirname(__file__)

def load():
    # 小鸟挥动翅膀的3个造型
    PLAYER_PATH = (
        assets_dir + '/assets/sprites/redbird-upflap.png',
        assets_dir + '/assets/sprites/redbird-midflap.png',
        assets_dir + '/assets/sprites/redbird-downflap.png'
    )

    # 游戏背景图，纯黑色是为了训练降低干扰
    BACKGROUND_PATH = assets_dir + '/assets/sprites/background-black.png'

    # 水管图片
    PIPE_PATH = assets_dir + '/assets/sprites/pipe-green.png'

    IMAGES, SOUNDS, HITMASKS = {}, {}, {}

    # 加载数字0~9的图片，类型是Surface图像
    IMAGES['numbers'] = (
        pygame.image.load(assets_dir + '/assets/sprites/0.png').convert_alpha(),
        pygame.image.load(assets_dir + '/assets/sprites/1.png').convert_alpha(),
        pygame.image.load(assets_dir + '/assets/sprites/2.png').convert_alpha(),
        pygame.image.load(assets_dir + '/assets/sprites/3.png').convert_alpha(),    # convert/conver_alpha是为了将图片转成绘制用的像素格式，提高绘制效率
        pygame.image.load(assets_dir + '/assets/sprites/4.png').convert_alpha(),
        pygame.image.load(assets_dir + '/assets/sprites/5.png').convert_alpha(),
        pygame.image.load(assets_dir + '/assets/sprites/6.png').convert_alpha(),
        pygame.image.load(assets_dir + '/assets/sprites/7.png').convert_alpha(),
        pygame.image.load(assets_dir + '/assets/sprites/8.png').convert_alpha(),
        pygame.image.load(assets_dir + '/assets/sprites/9.png').convert_alpha()
    )

    # 地面图片
    IMAGES['base'] = pygame.image.load(assets_dir + '/assets/sprites/base.png').convert_alpha()

    # 根据操作系统加载不同格式的声音文件
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    # 各种Sound对象
    SOUNDS['die']    = pygame.mixer.Sound(assets_dir + '/assets/audio/die' + soundExt)
    SOUNDS['hit']    = pygame.mixer.Sound(assets_dir + '/assets/audio/hit' + soundExt)
    SOUNDS['point']  = pygame.mixer.Sound(assets_dir + '/assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound(assets_dir + '/assets/audio/swoosh' + soundExt)
    SOUNDS['wing']   = pygame.mixer.Sound(assets_dir + '/assets/audio/wing' + soundExt)

    # 加载背景图片
    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()

    # 加载小鸟的3个姿态
    IMAGES['player'] = (
        pygame.image.load(PLAYER_PATH[0]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[1]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[2]).convert_alpha(),
    )

    # 加载水管图片，并反转180度产生上方的水管图片
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPE_PATH).convert_alpha(), 180),
        pygame.image.load(PIPE_PATH).convert_alpha(),
    )

    # 计算水管图片的bool掩码
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # 生成小鸟图片的bool掩码
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

    return IMAGES, SOUNDS, HITMASKS

# 生成图片的bool掩码矩阵，true表示对应像素位置不是透明的部分
def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))    # 像素点是RGBA，例如：(83, 56, 70, 255)，最后是透明度（0是透明，255是不透明）
    return mask
