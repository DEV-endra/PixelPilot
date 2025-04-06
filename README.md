# PixelPilot - AI Game Agent

PixelPilot is an AI-driven game agent designed to play a Car game ( [Fast Endless Police Car Chase](https://apps.microsoft.com/detail/9N7LW214VLR3?hl=en-us&gl=IN&ocid=pdpshare) )where the objective is to avoid police cars, collect cash, and avoid colliding with obstacles like stones. The agent is trained using reinforcement learning techniques with a Convolutional Neural Network (CNN) and Stable Baselines.

## Features
- **Reinforcement Learning**: Utilizes a PPO (Proximal Policy Optimization) algorithm and DQN(Deep Q-Networks).
- **Computer Vision**: Uses OpenCV and MSS for screen capturing and processing.
- **Text Recognition**: Implements PyTesseract for reading in-game information.
- **Custom Reward Function**: The agent is optimized to maximize game rewards based on strategic actions.
- **Gymnasium Integration**: The training environment follows the Gymnasium framework for RL training.

## Reward Function Design
The current reward system is:
- +2 for every frame survived (1 frame ~ 1 second)
- +10 for collecting cash 
- -100 for colliding with stones 
- -10 for hitting a police car 

## Setup Instructions
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- OpenCV
- NumPy
- MSS
- PyTesseract
- Gymnasium
- Stable Baselines3

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/pixelpilot.git
   cd pixelpilot
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Agent
To start training:
```bash
python train.py
```

To test the trained model:
```bash
python play.py --model models/pixelpilot_model.zip
```

## Known Issues
- Further tuning is required to balance reward functions for better decision-making.

## Contributors
- [Sriharshith](https://github.com/Sriharshith1863)
- [Rana Bharath](https://github.com/ranabharath)
- [Gnaneshwar](https://github.com/gnaneshwar-t)
- [Devendra Chand](https://github.com/DEV-endra)


https://github.com/user-attachments/assets/75a69d4e-5b5f-408b-a781-5cdf980a3148

