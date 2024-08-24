import torch
import random
import numpy as np
from collections import deque
from game import BlockBlast
from model import QNetwork, QTrainer
from helper import plot
import display
import time

MAX_MEMORY = 1_000_000
BATCH_SIZE = 10000
LR = 0.001

GRID_SIZE = 8
NUM_SHAPES = 3
MAX_ACTION = NUM_SHAPES * GRID_SIZE * GRID_SIZE

HIDDEN_SIZE = 256

class Agent:
    def __init__(self, input_size, output_size) -> None:
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = QNetwork(input_size, HIDDEN_SIZE, output_size)
        self.trainer = QTrainer(self.model, LR, self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        if random.randint(0, 1000) < self.epsilon:
            return random.randrange(MAX_ACTION - 1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            q_values = self.model(state0)
            return torch.argmax(q_values).item()

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    high_score = 0
    env = BlockBlast(GRID_SIZE)
    env.get_new_shapes()
    state_size = env.get_state().shape[0]

    agent = Agent(state_size, MAX_ACTION)
    while True:
        valid_action = False
        game_over = False

        i = 0

        while not valid_action:
            if i > 10:
                game_over = True
                break

            state = env.get_state()
            action = agent.get_action(state)
            if action > MAX_ACTION:
                print(f"action: {action} > MAX_ACTION: {MAX_ACTION}")
            next_state, reward, valid_action, score = env.step(action)
            game_over = env.is_game_over()

            if next_state is None:
                next_state = state


            agent.train_short_memory(state, action, reward, next_state, False)
            agent.remember(state, action, reward, next_state, False)

            i += 1

        #if valid_action:
        #    display.display_grid(score, env.grid)
        #    display.display_shapes(env.remaining_shapes)
        #else:
        #    print("Invalid Action")

        if game_over:
            env.reset()
            agent.n_games += 1

            agent.train_long_memory()

            if score > high_score:
                high_score = score

            print(f"Game #{agent.n_games}, Score: {score}, High Score: {high_score} Tries: {i}")

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            if len(plot_scores) > 100:
                plot_scores.pop(0)
                plot_mean_scores.pop(0)
            plot(plot_scores, plot_mean_scores)

        if agent.n_games > 1000:
            break

    print(f"Final High Score: {high_score}")

    while True:
        plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()