Thank you for checking out my project. This project is free of use but if you are going
to use it please credit me.
Github: https://github.com/GrantHen
LinkedIn: https://www.linkedin.com/in/granthendo/
Instagram: @granthendo56

This file includes:
- **Installation commands**
- **How to train and run the model**
- **Fixes for common errors**




🖐 Hand Gesture Recognition with Mediapipe & Scikit-Learn

This project uses **OpenCV**, **Mediapipe**, and **Scikit-Learn** to detect and classify hand gestures using a trained **RandomForestClassifier**. 





📌 **Setup Instructions MAC/LINUX**
Follow these steps to install dependencies, train the model, and run the gesture recognition system.

---

✅ **1. Install Python 3.10 (If Not Installed)**
Make sure you have **Python 3.10** installed. To check your Python version, run:

## ```sh
## python3 --version

If you don’t have Python 3.10, install it using Homebrew:
brew install python@3.10

Run the following command to install all necessary dependencies:
## /opt/homebrew/bin/python3.10 -m pip install --upgrade pip setuptools wheel
## /opt/homebrew/bin/python3.10 -m pip install opencv-python mediapipe scikit-learn pandas numpy joblib

Verify that all required libraries are installed:
## /opt/homebrew/bin/python3.10 -m pip list

Common issues:
FileNotFoundError: No such file or directory: 'gesture_classifier.pkl'
Solution: Ensure that you have run classifier.py to train and save the model.

ModuleNotFoundError: No module named 'cv2'
Solution: Ensure OpenCV is installed correctly:
/opt/homebrew/bin/python3.10 -m pip install opencv-python

ModuleNotFoundError: No module named 'mediapipe'
Solution: Reinstall Mediapipe:
/opt/homebrew/bin/python3.10 -m pip install mediapipe






📌 **Setup Instructions WINDOWS**
Follow these steps to install dependencies, train the model, and run the gesture recognition system.

---

## ✅ **1. Install Python 3.10 (If Not Installed)**
### 🔹 Check if Python is Installed:
Open **Command Prompt (cmd)** and run:

```sh
python --version

if Python 3.10 is not installed, download it from the official website:
🔗 Download Python 3.10 https://www.python.org/downloads/release/python-3100/

During installation, check the box that says: ✅ "Add Python to PATH"
After installation, verify:
## python --version

Once Python is installed, open Command Prompt (cmd) and install the dependencies:
## python -m pip install --upgrade pip setuptools wheel
## python -m pip install opencv-python mediapipe scikit-learn pandas numpy joblib

Verify that the required packages are installed:
## python -m pip list

Common Issues & Fixes
FileNotFoundError: No such file or directory: 'gesture_classifier.pkl'
Solution: Ensure that you have run classifier.py to train and save the model.

ModuleNotFoundError: No module named 'cv2'
Solution: Ensure OpenCV is installed correctly:
## python -m pip install opencv-python

ModuleNotFoundError: No module named 'mediapipe'
Solution: Reinstall Mediapipe:
## python -m pip install mediapipe





Check Python Interpreter Path
Sometimes Python scripts fail because the wrong interpreter is being used.

Check the Python Path in Terminal
Run:

## python -c "import sys; print(sys.executable)"
This should return something like:

C:\Users\YourUsername\AppData\Local\Programs\Python\Python310\python.exe
If it's not Python 3.10, you need to set the correct Python version in your environment.

