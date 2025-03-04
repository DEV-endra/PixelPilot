from mss import mss    # mss used for screen capture
import pydirectinput   # sending commands
import cv2             # open cv for frame processing
import numpy as np 
import pytesseract     # FOR OPTICAL CHARACTER RECOGNITION
from matplotlib import pyplot as plt
import time 
from gymnasium import spaces
from gymnasium import Env
from PIL import Image
from gym.spaces import Box, Discrete

def preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Increase contrast and sharpness
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
            sharpened, 255, 
            cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY_INV, 
            15, 8
        )

    # Morphological operations
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    return eroded

def press(delay, key):
    pydirectinput.keyDown(key)
    time.sleep(delay)
    pydirectinput.keyUp(key)
    
class WebGame(Env):
    def __init__(self):      # Setting up the environment and observation spaces
        super().__init__()   # for using the base class of gym
        
        self.observation_space = spaces.Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = spaces.Discrete(5)
        # Capture game frames
        self.cap = mss()
        self.game_location = {'top': 250, 'left': 650, 'width': 650, 'height':700}
        self.done_location = {'top':450, 'left': 500, 'width': 380, 'height':90}
    def step(self, action):
        action_map = { 
            0:'small_left',
            1: 'large_left', 
            2: 'small_right',
            3: 'large_right',
            4: 'no_op'
        }
        if action !=4:
            if(action==0):
                press(0.0000001,'left')
            elif(action==1):
                press(0.5,'left')
            elif(action==2):
                press(0.0000001,'right')
            else:
                press(0.5,'right')

        done, done_cap = self.get_done()   
        observation = self.get_observation()
        reward = 1 
        truncated = False
        info = {}
        return observation, reward, done,truncated, info
        
    def reset(self,seed=None,options=None):
        time.sleep(1)
        pydirectinput.click(x=960, y=540)
        pydirectinput.press('space')
        pydirectinput.press('space')
        return self.get_observation(),{}
        
    def render(self):
        cv2.imshow('Game', self.current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()
         
    def close(self): 
        cv2.destroyAllWindows()
    
    def get_observation(self):       # processing info from the frames 
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3].astype(np.uint8)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100,83))
        channel = np.reshape(resized, (1,83,100))
        return channel

    def get_done(self):
        done_cap = np.array(self.cap.grab(self.done_location)).astype(np.uint8)
        plt.imshow(self.cap.grab(self.done_location))

        done_cap=preprocessing(done_cap)

        done_strings = ['HIGH']
        done=False
        # if np.sum(done_cap) < 44300000:
        #     done = True
        done = False
        custom_config = r'--oem 3 --psm 6'
        res = pytesseract.image_to_string(done_cap, config=custom_config)[:4]
        
        # print(res)
        if res in done_strings:
            done = True
        return done, done_cap
