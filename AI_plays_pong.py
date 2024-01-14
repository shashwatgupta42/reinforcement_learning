from hard_pong_env import Env #CHANGE hard_pong_env TO pong_env FOR RUNNING EASY PONG
from RL_NN import Deep_Q_Learner
import numpy as np
import pygame 
import pickle
pygame.init()

FILENAME = 'saved_param/hard_pong_param.dat' #USE A saved_param/easy_pong_param.dat TO RUN EASY PONG
WIDTH, HEIGHT = 600, 600
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI plays Pong")

env = Env() #loading the pong environment
env.paddle.VEL = 40
env.ball.VEL = 50

def load_model(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    Q_Learner = Deep_Q_Learner(data['state_size'], data['nodes'], data['activations'])
    Q_Learner.train_original_param = data['trained_original_param']
    return Q_Learner

def play_pong(env, model):
    run = True
    state = env.reset()
    clock = pygame.time.Clock()
    while run:
        clock.tick(120)
        env.render(WIN)
        #state = (state[0], state[1], state[2], state[3], 0, 0, HEIGHT)
        state = np.expand_dims(np.array(state), axis=1)
        action = np.argmax(model.forward_pass(model.train_original_param, state))
        state, _, done = env.step(action)
        if done:
            state = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
    pygame.display.quit()
    pygame.quit()

if __name__ == '__main__':
    model = load_model(FILENAME)
    play_pong(env, model)