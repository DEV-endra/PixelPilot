#################################### TESTING THE ENVIRONMENT ####################################################
import os     
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
# Check Environment    
from stable_baselines3.common import env_checker

env_checker.check_env(env)  


#######################################  SETTING CALLBACKS  #####################################################
from stable_baselines3.common.callbacks import BaseCallback
class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1): 
        super(TrainAndLoggingCallback, self).__init__(verbose) 
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)  

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:  
            model_path = os.path.join(self.save_path, 'best_model_extended{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
    
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)


############################################### MODEL ############################################################

from stable_baselines3 import PPO 
from stable_baselines3.common.monitor import Monitor  
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
  
env = WebGame()
# First, wrap the environment in DummyVecEnv to make it compatible with VecFrameStack
env = DummyVecEnv([lambda: env])

# Now apply VecFrameStack to stack last 4 frames   
env = VecFrameStack(env, 4)
model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1)  # Reduced by 10xlearning_starts=1000)



###################################################### LOAD MODEL #################################################
model.load('train/best_model_extended1000')


####################################################### TRAIN #####################################################
model.learn(total_timesteps=21000, callback=callback)  



####################################################### EVALUATION ################################################
model.load('train/best_model_extended13000')   #CHOOSE ANY MODEL FROM THE SAVED ONES

for episode in range(1): 
    obs,_ = env.reset()
    done = False
    total_reward = 0
    while not done: 
        action,_ = model.predict(obs)
        obs, reward, done,_,info = env.step(int(action))
        time.sleep(0.01)
        total_reward += reward
    print('Total Reward for episode {} is {}'.format(episode, total_reward))
    time.sleep(2)