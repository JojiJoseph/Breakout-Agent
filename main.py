import gym
import cv2
import numpy as np

env = gym.make("Breakout-v4", render_mode="human")
print(env.unwrapped.get_action_meanings())

obs = env.reset()
env.unwrapped.ale.setRAM(57,255) # Set lives to 255
env.unwrapped.ale.setDifficulty(0)

action = 1
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

import time

pre_action = 2

video_writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 90, (160,210))

for i in range(10_0000):
    obs,_,_,done,_ = env.step(action)
    video_writer.write(obs[:,:,::-1])
    pre_action = action
    action = 1
    
    ball_roi = obs[94:188,8:152]
    bat_roi = obs[190:195,8:150]
    
    # Find ball location
    mask = cv2.inRange(ball_roi, (180,0,0), (220,74,74))
    ret,_,_,centroids_ball = cv2.connectedComponentsWithStats(mask)
    
    # Find bat location
    mask = cv2.inRange(bat_roi, (180,0,0), (220,74,74))
    ret,_,_,centroids_bat = cv2.connectedComponentsWithStats(mask)
    
    action = 1 # FIRE


    # Choose action based on locations of ball and bat
    if len(centroids_ball) > 1 and len(centroids_bat) > 1:
        cv2.circle(obs, np.int0((8+centroids_ball[1][0], 94 + centroids_ball[1][1])),4,(0,0,255),-1)
        cv2.circle(obs, np.int0((8+centroids_bat[1][0], 190 + centroids_bat[1][1])),4,(0,0,255),-1)
        x_ball = 8 + centroids_ball[1][0]
        x_bat = 8 + centroids_bat[1][0]
        
        if x_ball > x_bat + 0:
            action = 2
            # print("RIGHT")
        elif x_ball < x_bat - 0:
            action = 3
            # print("LEFT")
        else:
            action = pre_action
    
    cv2.imshow("Output", obs)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if done:
        break

video_writer.release()