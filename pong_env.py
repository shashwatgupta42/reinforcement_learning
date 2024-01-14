import pygame
import math
import random
import numpy as np
pygame.init()

WIDTH, HEIGHT = 600, 600

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (227, 75, 75)
BLUE = (71, 132, 237)

PADDLE_WIDTH, PADDLE_HEIGHT = 20, 100
BALL_RADIUS = 9

class Paddle:
    COLOR = BLUE
    VEL = 8
    X = WIDTH - 10 - PADDLE_WIDTH

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def random_y(self):
        y = random.randint(0, HEIGHT-self.height)
        return y
    
    def draw(self, win):
        pygame.draw.rect(
            win, self.COLOR, (self.x, self.y, self.width, self.height))
        
    def move(self, up):
        if up:
            self.y -= self.VEL
        else:
            self.y += self.VEL

    def reset(self):
        self.x = self.X
        self.y = self.random_y()

class Ball:
    COLOR = RED
    MAX_VEL = 10
    def __init__(self, radius):
        self.radius = radius

    def random_y(self):
        y = random.randint(0, HEIGHT-self.radius)
        return y
    
    def rand_velocity(self):
        self.x_vel = random.uniform(self.MAX_VEL//2.5, self.MAX_VEL//1.2)
        self.y_vel = random.choice([1, -1])*(self.MAX_VEL**2 - self.x_vel**2)**0.5

    def draw(self, win):
        pygame.draw.circle(win, self.COLOR, (self.x, self.y), self.radius)

    def move(self):
        self.x += self.x_vel
        self.y += self.y_vel

    def reset(self):
        self.x = self.radius
        self.y = self.random_y()
        self.rand_velocity()

class Env:
    OUTBOUND_PENALTY = -10
    MOVEMENT_PENALTY = 0.0
    MISS_MIN_PENALTY = -100
    HIT_MAX_REWARD = 150
    def __init__(self):
        self.ball = Ball(BALL_RADIUS)
        self.paddle = Paddle(PADDLE_WIDTH, PADDLE_HEIGHT)

    def reset(self):
        self.done = False
        self.ball.reset()
        self.paddle.reset()
        self.reward = 0
        state = self.return_state()
        return state

    def compute_ball_dirn(self):
        # returns the direction of ball in degrees wrt the horizontal axis
        ref_vector = (1, 0)
        ball_vector = (self.ball.x_vel, self.ball.y_vel)
        cos_theta = (ref_vector[0]*ball_vector[0] + ref_vector[1]*ball_vector[1])/(self.ball.MAX_VEL)
        theta = math.degrees(math.acos(cos_theta))
        if self.ball.y_vel > 0:
            theta *= -1
        return theta
    
    def return_state(self):
        return (self.ball.x, self.ball.y, self.compute_ball_dirn(), self.paddle.y)

    def render(self, win):
        win.fill(BLACK)
        self.paddle.draw(win)
        self.ball.draw(win)
        pygame.display.update()

    def take_action(self, action):
        if action == 1:
            if self.paddle.y - self.paddle.VEL >= 0:
                self.paddle.move(up=True)
                self.reward += self.MOVEMENT_PENALTY
            else:
                self.reward += self.OUTBOUND_PENALTY
        if action == 2:
            if self.paddle.y + self.paddle.height + self.paddle.VEL <= HEIGHT:
                self.paddle.move(up=False)
                self.reward += self.MOVEMENT_PENALTY
            else:
                self.reward += self.OUTBOUND_PENALTY

    def compute_miss_penalty(self):
        diff = math.fabs((self.paddle.y + self.paddle.height//2) - self.ball.y)
        return self.MISS_MIN_PENALTY*(1.0012**diff)

    def compute_hit_reward(self):
        diff = math.fabs((self.paddle.y + self.paddle.height//2) - self.ball.y)
        return self.HIT_MAX_REWARD*(0.992**diff)

    def handle_collisions(self):
        if self.ball.y+self.ball.radius >= HEIGHT or self.ball.y-self.ball.radius <= 0:
            self.ball.y_vel *= -1
        if self.ball.x-self.ball.radius <= 0:
            self.ball.x_vel *= -1

        if self.ball.x_vel > 0:
            if self.ball.x + self.ball.radius >= self.paddle.x:
                if self.ball.y >= self.paddle.y and self.ball.y <= self.paddle.y + self.paddle.height:
                    self.ball.x_vel *= -1
                    self.reward += self.compute_hit_reward()
        
        if self.ball.x + self.ball.radius >= WIDTH:
            self.reward += self.compute_miss_penalty()
            self.done = True

    def step(self, action):
        '''
        action=0 --> Nothing
        action=1 --> Up
        action=2 --> Down
        '''
        self.reward = 0
        self.take_action(action)
        self.ball.move()
        self.handle_collisions()

        next_state = self.return_state()
        return (next_state, self.reward, self.done)

if __name__ == '__main__':
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pong Env")
    run = True
    env = Env()
    state = env.reset()
    print(f"Initial State: {state} \n")
    clock = pygame.time.Clock()
    while run:
        clock.tick(60)
        env.render(WIN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action = 1
        elif keys[pygame.K_DOWN]:
            action = 2
        else:
            action = 0

        state, reward, done = env.step(action)
        print(f"State: {state}")
        print(f"Reward: {reward}")
        print("\n")
        if done:
            state = env.reset()
            print(f"Initial State: {state} \n")









