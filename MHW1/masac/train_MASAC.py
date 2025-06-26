import os
from my_masac import *
import my_masac.runner as my_masac
runner = my_masac.get_runner(
    method=['masac', 'masac'],                     # 使用QMIX方法
    env='mpe',                        # 多智能体强化学习环境类型
    env_id='simple_tag_v3',        # PettingZoo中的simple_tag环境
    is_test=False                  # 是否为测试模式
)
runner.run()
