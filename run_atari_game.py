# Modified by Sunbeam, Ravi (2023)
import tensorflow as tf
import cv2
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

def preprocess(image):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    width = 84
    height = 84
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame / 255.0

gril = tf.keras.models.load_model('gril_alien_adam_e50.h5')
env = gym.make("ALE/Alien-v5", render_mode="human")
observation, info = env.reset(seed=42)
# plt.imshow(observation)
# plt.show()
# observation = preprocess(observation)
# observation = np.expand_dims(observation, axis=-1)
# observation = np.expand_dims(observation, axis=0)
# print(observation.shape)

for _ in range(1000):
   observation = preprocess(observation)
   observation = np.expand_dims(observation, axis=-1)
   observation = np.expand_dims(observation, axis=0)
   action, gaze = gril.predict(observation)  # this is where you would insert your policy
   action = np.argmax(action)
   observation, reward, terminated, truncated, info = env.step(action) # ***tuple unpacking problem in this line using older gym versions
   # print(reward)
   print(action)
   if terminated or truncated:
      observation, info = env.reset()
env.close()
