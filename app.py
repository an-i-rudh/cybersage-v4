# app.py - CyberSage Logic
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import lightgbm as lgb
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.sparse import hstack, csr_matrix
import scipy.sparse as sp
import re
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# --- LOAD ARTIFACTS (Logic Unchanged) ---
lgb_model = lgb.Booster(model_file='bigvul_lgb_model.txt')
idf_code = np.load('idf_code.npy')
idf_commit = np.load('idf_commit.npy')
with open('vocab_code.pkl', 'rb') as f: vocab_code = pickle.load(f)
with open('vocab_commit.pkl', 'rb') as f: vocab_commit = pickle.load(f)
scaler_mean = np.load('scaler_mean.npy')
scaler_scale = np.load('scaler_scale.npy')
with open('meta.pkl', 'rb') as f: meta = pickle.load(f)

best_threshold = meta['best_threshold']
NUM_ALL = meta['num_features']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
cb_model = AutoModel.from_pretrained('microsoft/codebert-base').to(DEVICE)
for p in cb_model.parameters(): p.requires_grad = False
cb_model.eval()

# --- HELPERS ---
def strip_comments(code):
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//[^\n]*', '', code)
    return code

def get_embedding(text):
    with torch.no_grad():
        enc = tokenizer([text[:1500]], padding=True, truncation=True, max_length=256, return_tensors='pt').to(DEVICE)
        return cb_model(**enc).last_hidden_state[:, 0, :].cpu().float().numpy()

def manual_tfidf(text, vocab, idf, token_re):
    tokens = token_re.findall(text.lower())
    tf = {}
    for t in tokens:
        if t in vocab: tf[t] = tf.get(t, 0) + 1
    if not tf: return sp.csr_matrix((1, len(vocab)), dtype=np.float32)
    indices, data = [], []
    for term, count in tf.items():
        idx = vocab[term]
        data.append((1.0 + np.log(float(count))) * idf[idx])
        indices.append(idx)
    row = sp.csr_matrix((np.array(data, dtype=np.float32), (np.zeros(len(indices)), np.array(indices))), shape=(1, len(vocab)))
    norm = np.sqrt(row.multiply(row).sum())
    return row / norm if norm > 0 else row

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        func_before = data.get('func_before', '').strip()
        if not func_before: return jsonify({'error': 'Source code required'}), 400

        # 1. Feature Extraction
        x_code = manual_tfidf(func_before, vocab_code, idf_code, re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*'))
        x_commit = sp.csr_matrix((1, len(vocab_commit)), dtype=np.float32) # Empty since commit msg removed
        emb = get_embedding(func_before)
        num_vec = np.zeros((1, len(NUM_ALL)), dtype=np.float32)
        num_scaled = (num_vec - scaler_mean) / scaler_scale
        
        # 2. Prediction
        x_full = hstack([x_code, x_commit, csr_matrix(num_scaled), csr_matrix(emb)])
        prob = float(lgb_model.predict(x_full)[0])

        # 3. Pattern Auditing & Line Mapping
        clean_code = strip_comments(func_before)
        lines = func_before.split('\n')
        pattern_map = {
            'Buffer Overflow (strcpy/gets)': r'\b(strcpy|strcat|gets|sprintf)\b',
            'Insecure Memory Access': r'\b(memcpy|memmove|memset)\b',
            'Dynamic Memory Risk': r'\b(malloc|free|realloc|calloc)\b',
            'Format String Vulnerability': r'\b(printf|fprintf)\b\s*\(\s*[^"]+\s*[,)]'
        }
        
        detected_patterns = []
        highlight_lines = []
        
        for name, pat in pattern_map.items():
            if re.search(pat, clean_code, re.I):
                detected_patterns.append(name)
                # Identify which lines contain the pattern
                for i, line in enumerate(lines):
                    if re.search(pat, line, re.I):
                        highlight_lines.append(i + 1)

        # 4. Hybrid Logic Gate (Preserving your logic)
        if detected_patterns and prob < best_threshold:
            prob = best_threshold + 0.05
        elif not detected_patterns and prob >= best_threshold:
            prob = best_threshold - 0.01

        pred = int(prob >= best_threshold)

        return jsonify({
            'probability': round(prob, 4),
            'prediction': pred,
            'label': 'VULNERABLE' if pred == 1 else 'NOT VULNERABLE',
            'threshold': round(float(best_threshold), 2),
            'detected_patterns': list(set(detected_patterns)),
            'highlight_lines': list(set(highlight_lines))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)