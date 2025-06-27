import numpy as np
import matplotlib.pyplot as plt
#Q(s, a) += α * (r + γ * max Q(s', a') - Q(s, a))
start=(0,0)
end=(4,4)
obstacles=[(2,2),(3,1)]
bound_x=5
bound_y=5
actions = ['up', 'down', 'left', 'right']
Q=np.zeros((5,5,len(actions)))
action_dict={'up':(0,-1),'down':(0,1),'left':(-1,0),'right':(1,0)}
alpha = 0.1          # 学习率
gamma = 0.9          # 折扣因子
episodes=100
max_steps = 100      # 每回合最大步数
epsilon_start = 1.0      # 初始探索率
epsilon_min = 0.1       # 最低探索率
decay_rate = 0.01        # 衰减速度
def step(state,action):
    x,y=state
    dx,dy=action_dict[action]
    x=x+dx
    y=y+dy
    x=max(0,min(bound_x-1,x))
    y=max(0,min(bound_y-1,y))
    if (x,y) in obstacles:
        return state,-10
    elif (x,y) == end:
        return (x,y),10
    else:
        return (x,y),-1
def choose_action(state,episode):
    epsilon=max(epsilon_min,epsilon_start-decay_rate*episode)
    if np.random.uniform(0,1)<epsilon:
        return np.random.choice(range(len(actions)))
    else:
        return np.argmax(Q[state[0],state[1]])
    
rewards_per_episode = []

for episode in range(episodes):
    state=start
    total_reward = 0
    print(f"开始第{episode+1}轮训练")
    for _ in range(max_steps):
        action_id=choose_action(state,episode)
        action=actions[action_id]
        new_state,reward=step(state,action)
        old_value = Q[state[0], state[1], action_id]
        next_max = np.max(Q[new_state[0], new_state[1]])
        Q[state[0], state[1], action_id] = old_value + alpha * (reward + gamma * next_max - old_value)
        state=new_state
        total_reward+=reward
        if state==end:
            break
    rewards_per_episode.append(total_reward)
    
# 奖励收敛曲线

plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Rewards Over Episodes")
plt.savefig("rewards_curve.png", dpi=300)  # 保存为 PNG 图像，dpi 可选

def get_optimal_path():
    state = start
    path = [state]
    for _ in range(max_steps):
        action_idx = np.argmax(Q[state[0], state[1]])
        action = actions[action_idx]
        next_state, _ = step(state, action)
        if next_state == state:  # 卡住了（可能撞障碍）
            break
        path.append(next_state)
        if next_state == end:
            break
        state = next_state
    return path

path = get_optimal_path()

# 绘图显示路径
def visualize_path(path):
    grid = np.zeros((bound_x, bound_y), dtype=str)
    grid[:] = '.'
    for (x, y) in obstacles:
        grid[x][y] = 'X'
    for (x, y) in path:
        grid[x][y] = '*'
    sx, sy = start
    gx, gy = end
    grid[sx][sy] = 'S'
    grid[gx][gy] = 'G'

    for row in grid:
        print(' '.join(row))

    # 用 matplotlib 画图
    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, bound_x-0.5)
    ax.set_ylim(-0.5, bound_y-0.5)
    ax.set_xticks(np.arange(-0.5, bound_x, 1))
    ax.set_yticks(np.arange(-0.5, bound_y, 1))
    ax.grid(True)

    for (x, y) in obstacles:
        ax.add_patch(plt.Rectangle((y-0.5, bound_x-1-x-0.5), 1, 1, color='black'))
    for (x, y) in path:
        ax.plot(y, bound_x-1-x, marker='o', color='orange', linestyle='-')

    ax.text(start[1], bound_x-1-start[0], 'S', va='center', ha='center', color='black', fontsize=18)
    ax.text(end[1], bound_x-1-end[0], 'E', va='center', ha='center', color='black', fontsize=18)
    plt.title("Learned Path")
    plt.savefig("learned_path.png", dpi=300)

visualize_path(path)


