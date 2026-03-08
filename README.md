# 🔐 CyberSage: AI-Powered Vulnerability Detector

CyberSage is a hybrid security analysis tool that combines **CodeBERT** embeddings with a **LightGBM** classifier to identify vulnerabilities in C/C++ code.

## 🚀 Features
* **Hybrid Logic**: Combines deep learning context with regex-based pattern matching.
* **Line Highlighting**: Pinpoints the exact location of dangerous API calls.
* **Trained on BigVul**: Utilizes insights from over 150,000 real-world vulnerable functions.

## 🛠️ Setup
1. Clone the repo.
2. Install dependencies: `pip install flask flask-cors lightgbm transformers torch scipy`
3. Run the backend: `python app.py`
4. Open `index.html` in your browser.
