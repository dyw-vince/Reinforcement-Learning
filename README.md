# Reinforcement-Learning
**大三下强化学习作业集合**  

小作业1：使用强化学习算法，让机器人从5×5网格世界的左上角移动到左下角，找到最优路径的同时躲避障碍  

<table>
  <tr>
    <td><img src="https://github.com/dyw-vince/Reinforcement-Learning/blob/main/assignment1/learned_path.png?raw=true" width="300"></td>
    <td><img src="https://github.com/dyw-vince/Reinforcement-Learning/blob/main/assignment1/rewards_curve.png?raw=true" width="300"></td>
  </tr>
  <tr>
    <td align="center">Learned Path</td>
    <td align="center">Rewards Curve</td>
  </tr>
</table>

小作业2：使用DQN或基于策略方法(PG)解决cartpole问题  

<table>
  <tr>
    <td><img src="https://github.com/dyw-vince/Reinforcement-Learning/blob/main/assignment2/DQN_video.gif?raw=true" width="300"></td>
    <td><img src="https://github.com/dyw-vince/Reinforcement-Learning/blob/main/assignment2/PG_video.gif?raw=true" width="300"></td>
  </tr>
  <tr>
    <td align="center">2.DQN gif</td>
    <td align="center">4.PG gif</td>
  </tr>
</table>



小作业3：在 OpenAI Gym 的 CarRacing-v2 离散环境中实现DQN算法(能达到800分左右的结果)  

小作业4：使用PG解决cartpole问题，同2  

小作业5：在 OpenAI Gym 的 CarRacing-v2 连续环境中实现DDPG算法(能达到880分左右的结果)  

<table>
  <tr>
    <td><img src="https://github.com/dyw-vince/Reinforcement-Learning/blob/main/assignment3/DQN_video.gif?raw=true" width="300"></td>
    <td><img src="https://github.com/dyw-vince/Reinforcement-Learning/blob/main/assignment5/DDPG_video.gif?raw=true" width="300"></td>
  </tr>
  <tr>
    <td align="center">3.DQN gif</td>
    <td align="center">5.DDPG gif</td>
  </tr>
</table>


小作业6：使用基于GYM构建自定义环境“一维导航”，通过自定义环境构建A2C神经网络模型。  

<table>
  <tr>
    <td><img src="https://github.com/dyw-vince/Reinforcement-Learning/blob/main/assignment6/result.png?raw=true" width="300"></td>
  </tr>
  <tr>
    <td align="center">Rewards Curve</td>
  </tr>
</table>

小作业7：基于PettingZoo中的simple_spread环境实现MAPPO算法(平均能达到-6~-8的收敛效果) 

小作业8：基于PettingZoo中的simple_spread环境实现QMIX算法(平均能达到-14~-18的收敛效果)  

<table>
  <tr>
    <td><img src="https://github.com/dyw-vince/Reinforcement-Learning/blob/main/assignment7/result/gif/out4.gif?raw=true" width="300"></td>
    <td><img src="https://github.com/dyw-vince/Reinforcement-Learning/blob/main/assignment8/result/gif/out2.gif?raw=true" width="300"></td>
  </tr>
  <tr>
    <td align="center">7.MAPPO gif</td>
    <td align="center">8.QMIX gif</td>
  </tr>
</table>

小作业9：基于PettingZoo中的simple_adversary环境实现MASAC算法  

小作业10：基于PettingZoo中的simple_adversary环境实现MADDPG算法

<table>
  <tr>
    <td><img src="https://github.com/dyw-vince/Reinforcement-Learning/blob/main/assignment9/result.gif?raw=true" width="300"></td>
    <td><img src="https://github.com/dyw-vince/Reinforcement-Learning/blob/main/assignment10/result1.gif?raw=true" width="300"></td>
  </tr>
  <tr>
    <td align="center">9.MASAC gif</td>
    <td align="center">10.MADDPG gif</td>
  </tr>
</table>

中作业1：采用MASAC算法和MADDPG算法,使用GYM环境构建一场经典的森林狩猎合作游戏（simple-tag），至少3个猎手。其中游戏展示过程分别保存在maddpg_videos和masac_videos中

<table>
  <tr>
    <td><img src="https://github.com/dyw-vince/Reinforcement-Learning/blob/main/middle_assignment1/results/maddpg/maddpg_gif.gif?raw=true" width="300"></td>
    <td><img src="https://github.com/dyw-vince/Reinforcement-Learning/blob/main/middle_assignment1/results/masac/masac_gif.gif?raw=true" width="300"></td>
  </tr>
  <tr>
    <td align="center">MADDPG gif</td>
    <td align="center">MASAC gif</td>
  </tr>
</table>

中作业2：采用RND算法，学习通关游戏蒙特祖玛的复仇(MontezumaRevenge)(目前能达到3700分的得分)

<table>
  <tr>
    <td><img src=https://github.com/dyw-vince/Reinforcement-Learning/blob/main/middle_assignment2/results/MontezumaRevenge.gif?raw=true" width="300"></td>
    <td><img src=https://github.com/dyw-vince/Reinforcement-Learning/blob/main/middle_assignment2/results/reward_per_episode.png?raw=true" width="300"></td>
  </tr>
  <tr>
    <td align="center">Montezuma's Revenge gif</td>
    <td align="center">Montezuma's Revenge reward</td>
  </tr>
</table>

大作业:使用pygame库构建Flappy Bird游戏环境，模拟鸟类飞行、碰撞检测和得分机制。采用深度Q网络（DQN）算法，结合Keras深度学习框架，对游戏环境进行训练，使AI能够自主学习并优化游戏策略。我们分别实现了pytorch和tensorflow两种框架下的代码。

在tensorflow框架下的小鸟最高能获得236分

<table>
  <tr>
    <td><img src=https://github.com/dyw-vince/Reinforcement-Learning/blob/main/big_assignment/tensorflow/flappy_bird.gif?raw=true" width="300"></td>
    <td><img src=https://github.com/dyw-vince/Reinforcement-Learning/blob/main/big_assignment/tensorflow/training_curve_reward.png?raw=true" width="300"></td>
  </tr>
  <tr>
    <td align="center">tensorflow flappy_bird gif</td>
    <td align="center">tensorflow flappy_bird reward curve</td>
  </tr>
</table>

在pytorch框架下的小鸟最高能获得611分

<table>
  <tr>
    <td><img src=https://github.com/dyw-vince/Reinforcement-Learning/blob/main/big_assignment/pytorch/flappy_bird.gif?raw=true" width="300"></td>
    <td><img src=https://github.com/dyw-vince/Reinforcement-Learning/blob/main/big_assignment/pytorch/training_curve_reward.png?raw=true" width="300"></td>
  </tr>
  <tr>
    <td align="center">pytorch flappy_bird gif</td>
    <td align="center">pytorch flappy_bird reward curve</td>
  </tr>
</table>