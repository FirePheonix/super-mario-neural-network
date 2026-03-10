"""
Quick visual demo - watches a neural network play Mario in real time.
If a trained model exists, loads it. Otherwise uses a fresh random network.
"""
import pickle
import os
import cv2
from env_utils import make_env, preprocess
from model import NeuralNetwork
from config import MAX_STEPS_PER_EPISODE

model_path = "models/best_network.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        net = pickle.load(f)
    print("Loaded trained model from", model_path)
else:
    net = NeuralNetwork()
    print("No trained model found - using random network")

print("Opening Mario window... (Ctrl+C to quit)\n")

env = make_env()
result = env.reset()
obs = result[0] if isinstance(result, tuple) else result

total_reward = 0
for step in range(MAX_STEPS_PER_EPISODE):
    state = preprocess(obs)
    action = net.forward(state)
    step_result = env.step(action)

    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        done = terminated or truncated
    else:
        obs, reward, done, info = step_result

    frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (512, 480))
    cv2.imshow("Mario AI", frame)
    cv2.waitKey(1)

    total_reward += reward

    if done:
        print(f"Episode ended at step {step}")
        print(f"  x_pos:  {info.get('x_pos', '?')}")
        print(f"  score:  {info.get('score', '?')}")
        print(f"  reward: {total_reward:.1f}")
        break

env.close()
cv2.destroyAllWindows()
