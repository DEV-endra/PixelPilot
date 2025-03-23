pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" #OCR-TESERRACT PATH


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
    
def preprocessing2(image):
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Adaptive threshold to create a binary image
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 6
    )

    # Find contours to extract the largest connected component (digit)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area and keep the largest one (assuming it's the digit)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a mask and draw only the largest contour
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # Apply mask to remove unnecessary regions
        processed = cv2.bitwise_and(thresh, mask)
    else:
        processed = thresh  # Fallback if no contours are found

    return processed

    return cropped

def detect_objects(image):
    #print(image.shape)
    # Assuming the image shape is (height, width, 100)
    #image = image[:, :, :3]  # Take the first three channels
    # Select first three channels assuming image is in CHW format:
    image = image[:3, :, :]
    # Transpose to HWC format:
    image = np.transpose(image, (1, 2, 0))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #print(hsv)
    # Define HSV color ranges for different objects
    cash_lower = np.array([50, 100, 100])    # Green (Cash)
    cash_upper = np.array([90, 255, 255]) 

    police_lower = np.array([0, 0, 0])       # Black (Police Cars)
    police_upper = np.array([180, 255, 50]) 

    player_lower = np.array([0, 150, 100])   # Red (Player Car)
    player_upper = np.array([10, 255, 255])  

    stone_lower = np.array([15, 20, 150])    # Light brown/beige
    stone_upper = np.array([35, 100, 255]) 

    # Create masks
    cash_mask = cv2.inRange(hsv, cash_lower, cash_upper)
    police_mask = cv2.inRange(hsv, police_lower, police_upper)
    player_mask = cv2.inRange(hsv, player_lower, player_upper)
    stone_mask = cv2.inRange(hsv, stone_lower, stone_upper)
    #print(police_mask)
    
    # Count pixels
    cash_pixels = np.sum(cash_mask > 0)
    police_pixels = np.sum(police_mask > 0)
    player_pixels = np.sum(player_mask > 0)
    stone_pixels = np.sum(stone_mask > 0)

    return cash_pixels, police_pixels, stone_pixels

def press(delay, key):
    pydirectinput.keyDown(key)
    time.sleep(delay)
    pydirectinput.keyUp(key)
    
################################################ ENVIRONMENT CLASS ########################################################
class WebGame(Env):
    def __init__(self):      # Setting up the environment and observation spaces
        super().__init__()   # for using the base class of gym
        #self.observation_space = spaces.Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 83, 100), dtype=np.uint8)
        self.action_space = spaces.Discrete(9)
        self.current_score = 0  # Initialize score
        self.prev_score = 0  #

        # Capture game frames
        self.cap = mss()
        self.game_location = {'top': 250, 'left': 650, 'width': 650, 'height':700}
        self.done_location = {'top':440, 'left': 550, 'width': 380, 'height':90}
        self.score_location={'top':30, 'left': 865,'width': 240, 'height':100}

    def step(self, action):
        self.prev_score = self.current_score
        action_map = { 
            0:'left1',
            1: 'left2', 
            2: 'left3',
            3: 'left4',
            4: 'right1',
            5:'right2',
            6:'right3',
            7:'right4',
            8:'no_op'
        }
        if action != 8:
            if action == 0:
                press(0.1, 'left')  # left1
            elif action == 1:
                press(0.2, 'left')  # left2
            elif action == 2:
                press(0.3, 'left')  # left3
            elif action == 3:
                press(0.5, 'left')  # left4
            elif action == 4:
                press(0.1, 'right')  # right1
            elif action == 5:
                press(0.2, 'right')  # right2
            elif action == 6:
                press(0.3, 'right')  # right3
            else:
                press(0.5, 'right')  # right4
        
        done, done_cap = self.get_done()   
        observation = self.get_observation()

        self.current_score #need to be updated


        # Detect objects
        cash_pixels, police_pixels, stone_pixels = detect_objects(observation)

        # Define new reward function
        reward = 2
        if(self.current_score-self.prev_score >= 10):  #cash reward
            reward += self.current_score-self.prev_score
        if police_pixels > 20:  # If police car is detected
            reward -= 10
        if stone_pixels > 30:  # If stone is detected
            reward -= 200
        
        truncated = False
        info = {"score": self.current_score}  
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
    
    # def get_observation(self):       # processing info from the frames 
    #     raw = np.array(self.cap.grab(self.game_location))[:,:,:3].astype(np.uint8)
    #     gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    #     resized = cv2.resize(gray, (100,83))
    #     channel = np.reshape(resized, (1,83,100))
    #     return channel
    
    def get_observation(self):       
        raw = np.array(self.cap.grab(self.game_location))  # Capture screen
        raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR

        # Resize to (100, 83) with high-quality interpolation
        resized = cv2.resize(raw, (100, 83), interpolation=cv2.INTER_CUBIC)

        # Transpose to (Channels, Height, Width)
        return np.transpose(resized, (2, 0, 1))  # (3, 83, 100)
    
    def get_reward_observation(self):       
        raw = np.array(self.cap.grab(self.game_location))  # Capture screen
        raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR

        # Resize to (100, 83) with high-quality interpolation
        resized = cv2.resize(raw, (70, 50), interpolation=cv2.INTER_CUBIC)

        # Transpose to (Channels, Height, Width)
        return np.transpose(resized, (2, 0, 1))  # (3, 83, 100)
    
    def update_score(self):
        # 1) Capture raw image
        # raw_img = np.array(self.cap.grab(self.score_location)).astype(np.uint8)

        # plt.imshow(self.cap.grab(self.score_location))
        
        # # 2) Convert to grayscale
        # gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    
        # # 3) Apply thresholding (invert colors if needed)
        # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
        # 4) OCR using Tesseract
        done_cap = np.array(self.cap.grab(self.score_location)).astype(np.uint8)
        plt.imshow(self.cap.grab(self.score_location))

        done_cap=preprocessing(done_cap)
        
        plt.figure(figsize=(5,5))
        plt.imshow(done_cap, cmap='gray')
        plt.axis("off")  # Hide axes
        plt.title("Processed Image for OCR")
        plt.show()
        # custom_config = r'--oem 3 --psm 6'
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
        result = pytesseract.image_to_string(done_cap, config=custom_config)
        
        print(result)
        print("dev")
        return result
    
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