# PixelPilot
Creating an AI agent that autonomously learns to play games using deep reinforcement learning with Stable Baselines and Gymnasium


## Prerequisites  
This project requires **Tesseract-OCR** to be installed on your system, as `pytesseract` is just a wrapper for it.  

### Installation Instructions:  
- **Windows:**  
  Download the installer from [Tesseract GitHub](https://github.com/ub-mannheim/tesseract/wiki) and add the installation path (e.g., `C:\Program Files\Tesseract-OCR`) to your system's PATH environment variable.  

c 
    ```bash
    sudo apt update
    sudo apt install tesseract-ocr
    ```

- **macOS (using Homebrew):**  
    ```bash
    brew install tesseract
    ```
After installing, verify by running:  
```bash
tesseract --version
```

- **Python Dependencies Installation**
Ensure you have Python installed (recommended: Python 3.7+). You can check your version with:
```bash
python --version
```

- **(Optional) Create a Virtual Environment**
It is recommended to create a virtual environment before installing dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

- **Install Required Packages**
Run the following commands to install the necessary dependencies:
```bash
pip install stable-baselines3[extra] protobuf==3.20.*
pip install mss pydirectinput pytesseract
```

### **Package Descriptions**  

| Package                     | Description |
|-----------------------------|------------|
| **stable-baselines3[extra]** | Reinforcement learning framework with additional dependencies. |
| **protobuf==3.20.***        | Ensures compatibility with libraries requiring older protobuf versions. |
| **mss**                     | Captures screen images efficiently for real-time processing. |
| **pydirectinput**           | Allows simulating keyboard and mouse input in Windows environments. |
| **pytesseract**             | Provides Optical Character Recognition (OCR) capabilities using Tesseract. |


###################################################### RESULTS #############################################################
https://github.com/user-attachments/assets/75a69d4e-5b5f-408b-a781-5cdf980a3148

