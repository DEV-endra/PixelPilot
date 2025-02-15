# PixelPilot
Creating an AI agent that autonomously learns to play games using deep reinforcement learning with Stable Baselines and Gymnasium


## Prerequisites  
This project requires **Tesseract-OCR** to be installed on your system, as `pytesseract` is just a wrapper for it.  

### Installation Instructions:  
- **Windows:**  
  Download the installer from [Tesseract GitHub](https://github.com/ub-mannheim/tesseract/wiki) and add the installation path (e.g., `C:\Program Files\Tesseract-OCR`) to your system's PATH environment variable.  

- **Linux (Ubuntu/Debian):**  
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
