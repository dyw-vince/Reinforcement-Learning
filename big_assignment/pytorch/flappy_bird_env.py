# flappy_bird_env.py
import pygame
import random
import numpy as np

SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
PIPE_GAP_SIZE = 100  # 上下管道的间距
PIPE_VELOCITY = -4

class FlappyBirdEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('PyTorch Flappy Bird')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 25)

        self.background = pygame.image.load('assets/sprites/background-day.png').convert()
        self.bird_img = pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha()
        self.pipe_img = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.base_img = pygame.image.load('assets/sprites/base.png').convert()
        
        self.action_space = [0, 1] # 0: 不动, 1: 飞
        self.state_space_dim = 4 # 状态维度: [鸟y, 鸟y速度, 下一个管道x, 下一个管道y]
        
    def reset(self):
        self.bird_x = 50
        self.bird_y = int(SCREEN_HEIGHT / 2)
        self.bird_velocity = 0
        self.pipes = [self._get_random_pipe()]
        self.score = 0
        self.base_x = 0
        return self._get_state()

    def step(self, action):
        pygame.event.pump()
        reward = 0.1 # 存活奖励
        done = False

        if action == 1:
            self.bird_velocity = 8 # 向上飞

        self.bird_velocity -= 1 # 重力
        self.bird_y -= self.bird_velocity

        # 移动管道
        for pipe in self.pipes:
            pipe['x'] += PIPE_VELOCITY

        # 生成新管道
        if 0 < self.pipes[0]['x'] < 5:
            self.pipes.append(self._get_random_pipe())

        # 移除屏幕外的管道
        if self.pipes[0]['x'] < -self.pipe_img.get_width():
            self.pipes.pop(0)
            reward = 1 # 通过管道奖励

        # 碰撞检测
        if self._check_collision():
            reward = -1 # 死亡惩罚
            done = True
        
        # 更新分数
        if not done:
            self.score += reward

        return self._get_state(), reward, done, self.score

    def render(self):
        self.screen.blit(self.background, (0, 0))

        for pipe in self.pipes:
            self.screen.blit(self.pipe_img, (pipe['x'], pipe['y_upper']))
            self.screen.blit(pygame.transform.flip(self.pipe_img, False, True), (pipe['x'], pipe['y_lower']))

        self.screen.blit(self.base_img, (self.base_x, SCREEN_HEIGHT * 0.79))
        self.base_x = (self.base_x - 4) % (SCREEN_WIDTH - self.base_img.get_width())

        self.screen.blit(self.bird_img, (self.bird_x, self.bird_y))
        
        score_text = self.font.render(f"Score: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.update()
        self.clock.tick(30)

    def _get_random_pipe(self):
        pipe_height = self.pipe_img.get_height()
        gap_y = random.randrange(int(SCREEN_HEIGHT * 0.3), int(SCREEN_HEIGHT * 0.7 - PIPE_GAP_SIZE))
        return {'x': SCREEN_WIDTH + 10, 'y_upper': gap_y - pipe_height, 'y_lower': gap_y + PIPE_GAP_SIZE}

    def _check_collision(self):
        if self.bird_y > SCREEN_HEIGHT * 0.79 - self.bird_img.get_height() or self.bird_y < 0:
            return True

        bird_rect = pygame.Rect(self.bird_x, self.bird_y, self.bird_img.get_width(), self.bird_img.get_height())
        for pipe in self.pipes:
            pipe_upper_rect = pygame.Rect(pipe['x'], pipe['y_upper'], self.pipe_img.get_width(), self.pipe_img.get_height())
            pipe_lower_rect = pygame.Rect(pipe['x'], pipe['y_lower'], self.pipe_img.get_width(), self.pipe_img.get_height())
            if bird_rect.colliderect(pipe_upper_rect) or bird_rect.colliderect(pipe_lower_rect):
                return True
        return False
        
    def _get_state(self):
        next_pipe = self.pipes[0]
        if self.bird_x > next_pipe['x'] + self.pipe_img.get_width():
            next_pipe = self.pipes[1]

        state = [
            (self.bird_y - (next_pipe['y_lower'] - PIPE_GAP_SIZE / 2)) / SCREEN_HEIGHT,
            self.bird_velocity / 10,
            (next_pipe['x'] - self.bird_x) / SCREEN_WIDTH,
            (next_pipe['y_lower'] - PIPE_GAP_SIZE / 2) / SCREEN_HEIGHT
        ]
        return np.array(state, dtype=np.float32)
    
    def close(self):
        pygame.quit()