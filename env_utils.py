import gym_super_mario_bros
import cv2
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from config import MAX_STEPS_PER_EPISODE

def make_env():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env

def preprocess(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (32, 32))
    return resized.flatten() / 255.0

def evaluate_network(network, render=False, window_name="Mario"):
    env = make_env()
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result

    total_reward = 0.0
    max_x = 0
    steps_taken = 0

    for _ in range(MAX_STEPS_PER_EPISODE):
        state = preprocess(obs)
        action = network.forward(state)
        step_result = env.step(action)

        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result

        # Display using OpenCV — obs IS the game frame (RGB 240x256)
        if render:
            frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (300, 270))
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

        total_reward += reward
        steps_taken += 1

        if "x_pos" in info:
            max_x = max(max_x, info["x_pos"])

        if done:
            break

    env.close()
    if render:
        cv2.destroyWindow(window_name)

    # Fitness: x progress + rewards - gentle time penalty (don't punish careful play)
    time_penalty = 0.01 * steps_taken
    fitness = max_x + 0.1 * total_reward - time_penalty
    return fitness