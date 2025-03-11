from flask import Flask, render_template, request, jsonify, Response
import gymnasium as gym
import base64
import numpy as np
from PIL import Image
import io
import json
import time

app = Flask(__name__)

# ゲーム環境の状態を保持するグローバル変数
sessions = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_game', methods=['POST'])
def start_game():
    session_id = str(time.time())  # 単純なセッションID
    env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array')
    observation, info = env.reset()
    
    # セッション情報を保存
    sessions[session_id] = {
        'env': env,
        'observation': observation,
        'score': 0,
        'done': False
    }
    
    # 画像をBase64エンコード
    img = Image.fromarray(observation)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return jsonify({
        'session_id': session_id,
        'image': img_str,
        'score': 0,
        'game_over': False
    })

@app.route('/take_action', methods=['POST'])
def take_action():
    data = request.json
    session_id = data.get('session_id')
    action = data.get('action')
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    session = sessions[session_id]
    
    if session['done']:
        return jsonify({
            'image': '',
            'score': session['score'],
            'game_over': True
        })
    
    # アクションを実行
    observation, reward, terminated, truncated, info = session['env'].step(action)
    
    # セッション情報を更新
    session['observation'] = observation
    session['score'] += reward
    session['done'] = terminated or truncated
    
    # 画像をBase64エンコード
    img = Image.fromarray(observation)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # ゲームオーバーの場合はリソースを解放
    if session['done']:
        session['env'].close()
    
    return jsonify({
        'image': img_str,
        'score': session['score'],
        'game_over': session['done']
    })

@app.route('/cleanup', methods=['POST'])
def cleanup():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id in sessions:
        if 'env' in sessions[session_id]:
            sessions[session_id]['env'].close()
        del sessions[session_id]
    
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
