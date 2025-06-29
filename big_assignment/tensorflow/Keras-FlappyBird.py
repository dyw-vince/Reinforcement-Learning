import pygame
import numpy as np
import cv2
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import os
import imageio 

# --- 游戏环境部分 (Pygame) ---

# 定义游戏常量
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
PIPE_WIDTH = 52
PIPE_HEIGHT = 320
PIPE_GAP_SIZE = 120
BIRD_WIDTH = 34
BIRD_HEIGHT = 24
GROUND_Y = SCREEN_HEIGHT * 0.79
PIPE_SPEED = -4
GRAVITY = 1
FLAP_STRENGTH = -9

class FlappyBirdEnv:
    def __init__(self, render=True):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Keras Flappy Bird')
        self.clock = pygame.time.Clock()
        self.render_mode = render

        try:
            self.background_img = pygame.image.load('assets/sprites/background-black.png').convert()
            self.bird_img = pygame.image.load('assets/sprites/redbird-midflap.png').convert_alpha()
            self.pipe_img = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
            self.ground_img = pygame.image.load('assets/sprites/base.png').convert()
        except (pygame.error, FileNotFoundError):
            print("警告: 资源图片加载失败，将使用纯色方块代替。")
            self.background_img = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT)); self.background_img.fill((0, 0, 0))
            self.bird_img = pygame.Surface((BIRD_WIDTH, BIRD_HEIGHT)); self.bird_img.fill((255, 0, 0))
            self.pipe_img = pygame.Surface((PIPE_WIDTH, PIPE_HEIGHT)); self.pipe_img.fill((0, 200, 0))
            self.ground_img = pygame.Surface((336, 112)); self.ground_img.fill((220, 200, 120))


        self.pipe_img_flipped = pygame.transform.flip(self.pipe_img, False, True)
        self.ground_x = 0
        self.reset()

    def reset(self):
        self.bird_y = int(SCREEN_HEIGHT / 2)
        self.bird_velocity = 0
        self.pipes = []
        pair1 = self._generate_new_pipes(x=SCREEN_WIDTH * 1.0)
        pair2 = self._generate_new_pipes(x=SCREEN_WIDTH * 1.0 + (SCREEN_WIDTH + PIPE_WIDTH) / 2)
        self.pipes.extend(pair1)
        self.pipes.extend(pair2)
        self.score = 0
        self.game_over = False
        return self._get_processed_state()

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                pygame.quit()
                quit()

        if action == 1:
            self.bird_velocity = FLAP_STRENGTH

        self.bird_velocity += GRAVITY
        self.bird_y += self.bird_velocity
        self._move_pipes()
        
        self.ground_x += PIPE_SPEED
        if self.ground_x <= -self.ground_img.get_width() + SCREEN_WIDTH:
            self.ground_x = 0

        self.game_over = self._check_collision()

        reward = 0.1 
        if self._passed_pipe():
            self.score += 1
            reward = 1.0 
        if self.game_over:
            reward = -1.0 

        next_state = self._get_processed_state()
        self.clock.tick(60)
        self._draw_frame() 
        return next_state, reward, self.game_over, self.score
    
    def get_render_frame(self):
        return pygame.surfarray.array3d(self.screen).transpose([1, 0, 2])

    def _generate_new_pipes(self, x):
        gap_y = random.randrange(int(GROUND_Y * 0.3), int(GROUND_Y * 0.7))
        upper_pipe = {'x': x, 'y': gap_y - PIPE_HEIGHT, 'passed': False}
        lower_pipe = {'x': x, 'y': gap_y + PIPE_GAP_SIZE, 'passed': False}
        return [upper_pipe, lower_pipe]

    def _move_pipes(self):
        for pipe in self.pipes:
            pipe['x'] += PIPE_SPEED
        if self.pipes[0]['x'] < -PIPE_WIDTH:
            self.pipes.pop(0)
            self.pipes.pop(0)
            new_x = self.pipes[-1]['x'] + (SCREEN_WIDTH + PIPE_WIDTH) / 2
            new_pipes = self._generate_new_pipes(x=new_x)
            self.pipes.extend(new_pipes)
            
    def _check_collision(self):
        bird_rect = pygame.Rect(SCREEN_WIDTH / 4, self.bird_y, BIRD_WIDTH, BIRD_HEIGHT)
        if self.bird_y > GROUND_Y - BIRD_HEIGHT or self.bird_y < 0:
            return True
        for pipe in self.pipes:
            pipe_rect = pygame.Rect(pipe['x'], pipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            if bird_rect.colliderect(pipe_rect):
                return True
        return False

    def _passed_pipe(self):
        bird_mid_pos = SCREEN_WIDTH / 4 + BIRD_WIDTH / 2
        for pipe in self.pipes:
            if not pipe.get('passed', False):
                pipe_mid_pos = pipe['x'] + PIPE_WIDTH / 2
                if pipe_mid_pos < bird_mid_pos:
                    pipe['passed'] = True
                    for p in self.pipes:
                        if p['x'] == pipe['x'] and p is not pipe:
                            p['passed'] = True
                            break
                    return True
        return False
    
    def _get_processed_state(self):
        self._draw_frame() 
        image_data = pygame.surfarray.array3d(self.screen)
        image_data = cv2.cvtColor(cv2.resize(image_data, (84, 84)), cv2.COLOR_BGR2GRAY)
        image_data = image_data / 255.0
        return np.reshape(image_data, (84, 84, 1))

    def _draw_frame(self):
        self.screen.blit(self.background_img, (0, 0))
        for pipe in self.pipes:
            if pipe['y'] < 0:
                self.screen.blit(self.pipe_img_flipped, (pipe['x'], pipe['y']))
            else:
                self.screen.blit(self.pipe_img, (pipe['x'], pipe['y']))
        self.screen.blit(self.ground_img, (self.ground_x, GROUND_Y))
        self.screen.blit(self.bird_img, (SCREEN_WIDTH / 4, self.bird_y))
        score_text = pygame.font.Font(None, 40).render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        if self.render_mode:
            pygame.display.update()

# --- DQN 智能体部分 (Keras) ---
class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        
        # --- 核心优化: 调整超参数以促进更有效的学习 ---
        self.gamma = 0.99  # 提高折扣因子，让AI更有远见
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995  # 显著减慢衰减速度，给予AI更多探索时间
        self.learning_rate = 1e-4  # 降低学习率，使训练更稳定
        
        self.update_target_freq = 5
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        input_layer = Input(shape=self.state_shape)
        conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_layer)
        conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
        conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)
        flatten = Flatten()(conv3)
        dense1 = Dense(512, activation='relu')(flatten)
        output_layer = Dense(self.action_size, activation='linear')(dense1)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        print("...目标网络权重已更新...")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, exploit_only=False):
        if exploit_only or np.random.rand() > self.epsilon:
            act_values = self.model.predict(state, verbose=0)
            return np.argmax(act_values[0])
        return random.randrange(self.action_size)

    def replay(self, batch_size):
        if len(self.memory) < batch_size * 10:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        
        target_q_next = self.target_model.predict(next_states, verbose=0)
        targets = rewards + self.gamma * np.amax(target_q_next, axis=1) * (1 - dones)
        
        current_q_values = self.model.predict(states, verbose=0)
        for i, action in enumerate(actions):
            current_q_values[i][action] = targets[i]
            
        self.model.fit(states, current_q_values, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def record_and_save_gif(agent_to_record, episode_number):
    print(f"\n--- Episode {episode_number}: 正在录制GIF动画... ---")
    frames = [] 
    record_env = FlappyBirdEnv(render=True)
    
    initial_frame = record_env.reset()
    current_state = np.stack([initial_frame] * 4, axis=2)
    current_state = np.reshape(current_state, [1, 84, 84, 4])
    done = False
    
    while not done:
        action = agent_to_record.act(current_state, exploit_only=True) 
        next_frame, _, done, score = record_env.step(action)
        rgb_frame = record_env.get_render_frame()
        frames.append(rgb_frame)
        next_frame = np.reshape(next_frame, [1, 84, 84, 1])
        next_state = np.append(next_frame, current_state[:, :, :, :3], axis=3)
        current_state = next_state
        if len(frames) > 1000:
            print("达到最大录制帧数，停止录制。")
            break

    print(f"录制完成! AI得分为: {score}")
    gif_path = f"flappy_bird_episode_{episode_number}.gif"
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"动画已成功保存到: {gif_path}\n")

if __name__ == "__main__":
    # --- 核心优化: 增加训练回合数，给予AI足够的学习时间 ---
    EPISODES = 2000 
    BATCH_SIZE = 64
    STATE_SHAPE = (84, 84, 4)
    ACTION_SIZE = 2
    
    env = FlappyBirdEnv(render=False) 
    agent = DQNAgent(state_shape=STATE_SHAPE, action_size=ACTION_SIZE)
    model_path = "flappybird_dqn.weights.h5"
    if os.path.exists(model_path):
        print(f"加载已存在的模型: {model_path}")
        agent.load(model_path)
        agent.epsilon = agent.epsilon_min

    for e in range(EPISODES):
        initial_frame = env.reset()
        current_state = np.stack([initial_frame] * 4, axis=2)
        current_state = np.reshape(current_state, [1, 84, 84, 4])
        done = False
        
        while not done:
            action = agent.act(current_state)
            next_frame, reward, done, score = env.step(action)
            next_frame = np.reshape(next_frame, [1, 84, 84, 1])
            next_state = np.append(next_frame, current_state[:, :, :, :3], axis=3)
            agent.remember(current_state, action, reward, next_state, done)
            current_state = next_state
            agent.replay(BATCH_SIZE)
            if done:
                if (e + 1) % agent.update_target_freq == 0:
                    agent.update_target_model()
                print(f"Episode: {e+1}/{EPISODES}, Score: {score}, Epsilon: {agent.epsilon:.4f}")
                break
        
        if (e + 1) % 50 == 0 and e > 0:
            agent.save(model_path)
            print(f"模型已保存在 {model_path}")
            record_and_save_gif(agent, e + 1)

    print("\n所有训练回合完成!")
    record_and_save_gif(agent, "final")
    
    pygame.quit()
