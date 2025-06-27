import os
from my_masac import *
import my_masac.runner as my_masac
runner =my_masac.get_runner(
    method=['masac','masac'],                     # 使用MASAC方法
    env='mpe',                        # 多智能体强化学习环境类型
    env_id='simple_adversary_v3',        # PettingZoo中的simple_adversary环境
    is_test=True            # 是否为测试模式
)
runner.run()