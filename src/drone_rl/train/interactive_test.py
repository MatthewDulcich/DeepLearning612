from gymnasium.utils.play import play
import gymnasium as gym
import pygame
import flycraft

env = gym.make('FlyCraft-v0', render_mode="human")
key_to_action = {
    (pygame.K_LEFT,): 0,  # Map left arrow key to action 0 (move left)
    (pygame.K_RIGHT,): 1, # Map right arrow key to action 1 (move right)
    (pygame.K_UP,): 2,    # Map up arrow key to action 2 (move up)
    (pygame.K_DOWN,): 3,  # Map down arrow key to action 3 (move down)
}
play(env, keys_to_action=key_to_action)
