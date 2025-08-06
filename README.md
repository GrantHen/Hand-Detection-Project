# Hand-Detection-Project
Thank you for checking out my project. This project is free of use but if you are going
to use it please credit me.
Github: https://github.com/GrantHen
LinkedIn: https://www.linkedin.com/in/granthendo/


Hand Gesture Recognition with Mediapipe & Scikit-Learn

This project uses **OpenCV**, **Mediapipe**, and **Scikit-Learn** to detect and classify hand gestures using a trained **RandomForestClassifier**. 

**Setup Instructions MAC/LINUX**
Follow these steps to install dependencies, train the model, and run the gesture recognition system.

---
 **1. Install Python 3.10 (If Not Installed)**
Make sure you have **Python 3.10** installed. To check your Python version, run:

```bash
## python3 --version
```

Run the following command to install all necessary dependencies:
```bash
pip install --upgrade pip setuptools wheel
pip install opencv-python mediapipe scikit-learn pandas numpy joblib
```

Verify that all required libraries are installed:
```bash
pip list
```

issues ive ran into:
FileNotFoundError: No such file or directory: 'gesture_classifier.pkl'
Solution: Ensure that you have run classifier.py to train and save the model.

ModuleNotFoundError: No module named 'cv2'
Solution: Ensure OpenCV is installed correctly:
```bash 
pip install opencv-python
```

ModuleNotFoundError: No module named 'mediapipe'
Solution: Reinstall Mediapipe:
```bash
pip install mediapipe
```






**Setup Instructions WINDOWS**
Follow these steps to install dependencies, train the model, and run the gesture recognition system.

---

## **1. Install Python 3.10 (If Not Installed)**
### 🔹 Check if Python is Installed:
Open **Command Prompt (cmd)** and run:

```bash
python --version
```

Once Python is installed, open Command Prompt (cmd) and install the dependencies:
```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install opencv-python mediapipe scikit-learn pandas numpy joblib
```


Common Issues & Fixes
FileNotFoundError: No such file or directory: 'gesture_classifier.pkl'
Solution: Ensure that you have run classifier.py to train and save the model.

ModuleNotFoundError: No module named 'cv2'
Solution: Ensure OpenCV is installed correctly:
```bash
python -m pip install opencv-python
```

ModuleNotFoundError: No module named 'mediapipe'
Solution: Reinstall Mediapipe:
```bash
python -m pip install mediapipe
```




Check Python Interpreter Path
Sometimes Python scripts fail because the wrong interpreter is being used.

Check the Python Path in Terminal
Run:

```bash
python -c "import sys; print(sys.executable)"
```
