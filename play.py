import gym
import numpy as np
import cv2
import gym.spaces as spaces
from collections import deque

class Wrapper(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.env = env
        # self.env.unwrapped.ale.setRAM(57,2) # Set lives to 255
        print(env.action_space)
        print(env.observation_space)
        self.obs_q = deque([0]*12, maxlen=12)
        self.observation_space = spaces.Box(-1, 1, (12,))
    def reset(self):
        obs = self.env.reset()
        ball_roi = obs[94:188,8:154]
        bat_roi = obs[190:195,8:152]
        # Find ball location
        mask = cv2.inRange(ball_roi, (180,0,0), (220,74,74))
        ret,_,_,centroids_ball = cv2.connectedComponentsWithStats(mask)
        
        # Find bat location
        mask = cv2.inRange(bat_roi, (180,0,0), (220,74,74))
        ret,_,stats,centroids_bat = cv2.connectedComponentsWithStats(mask)
        x_ball = -1
        x_bat = -1
        y_ball = -1
        if len(centroids_ball) > 1 and len(centroids_bat) > 1:
        # cv2.circle(obs, np.int0((8+centroids_ball[1][0], 94 + centroids_ball[1][1])),4,(0,0,255),-1)
        # cv2.circle(obs, np.int0((8+centroids_bat[1][0], 190 + centroids_bat[1][1])),4,(0,0,255),-1)
            x_ball = 8 + centroids_ball[1][0]
            y_ball = 94 + centroids_ball[1][1]
            x_bat = 8 + centroids_bat[1][0]
            x_ball = (x_ball-80) / 80
            x_bat = (x_bat - 80) /80
            y_ball = (y_ball - 105)/105
        self.obs_q.extend([x_ball, y_ball, x_bat])
        obs = np.array(self.obs_q)
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        ball_roi = obs[94:188,8:154]
        bat_roi = obs[190:195,8:152]
        
        # Find ball location
        mask = cv2.inRange(ball_roi, (180,0,0), (220,74,74))
        ret,_,_,centroids_ball = cv2.connectedComponentsWithStats(mask)
        
        # Find bat location
        mask = cv2.inRange(bat_roi, (180,0,0), (220,74,74))
        ret,_,stats,centroids_bat = cv2.connectedComponentsWithStats(mask)
        x_ball = -1
        x_bat = -1
        y_ball = -1
        if len(centroids_ball) > 1 and len(centroids_bat) > 1:
        # cv2.circle(obs, np.int0((8+centroids_ball[1][0], 94 + centroids_ball[1][1])),4,(0,0,255),-1)
        # cv2.circle(obs, np.int0((8+centroids_bat[1][0], 190 + centroids_bat[1][1])),4,(0,0,255),-1)
            x_ball = 8 + centroids_ball[1][0]
            y_ball = 94 + centroids_ball[1][1]
            x_bat = 8 + centroids_bat[1][0]
            x_ball = (x_ball-80) / 80
            x_bat = (x_bat - 80) /80
            y_ball = (y_ball - 105)/105
        self.obs_q.extend([x_ball, y_ball, x_bat])
        obs = np.array(self.obs_q)
        return obs, rew, done, info

env = gym.make("Breakout-v0", render_mode="human")
env = Wrapper(env)
done = False
obs = env.reset()
from stable_baselines3 import PPO, DQN
model = PPO.load("logs/best_model.zip")
while not done:
    # env.render()
    action = model.predict(obs)
    obs, rew, done, info = env.step(action[0])
