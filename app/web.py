import os
import threading
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, Response


MODEL_PATH = r"D:\emotion-recognition\models\final_fer_model.h5"
print(f"Checking model at: {os.path.abspath(MODEL_PATH)}")

model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")

EMOTIONS = ['angry', 'happy', 'neutral', 'sad', 'surprise']

EMOTION_META = {
    'angry':    {'color': '#ef4444', 'vn': 'Tức giận',
                 'msg': 'Hãy hít thở sâu và bình tĩnh lại nhé. Mọi chuyện rồi sẽ ổn thôi!'},
    'happy':    {'color': '#f59e0b', 'vn': 'Vui vẻ',
                 'msg': 'Tuyệt vời! Nụ cười của bạn thật đẹp. Hãy lan tỏa niềm vui đó nhé!'},
    'neutral':  {'color': '#64748b', 'vn': 'Bình thường',
                 'msg': 'Bạn đang ở trạng thái bình ổn. Đây là thời điểm tốt để tập trung!'},
    'sad':      {'color': '#3b82f6', 'vn': 'Buồn bã',
                 'msg': 'Đừng lo, mọi cơn buồn đều sẽ qua. Hãy tự thưởng cho mình điều gì đó dễ chịu nhé!'},
    'surprise': {'color': '#a855f7', 'vn': 'Ngạc nhiên',
                 'msg': 'Ồ, có chuyện gì bất ngờ vậy? Hi vọng đó là một điều thú vị!'},
}

app = Flask(__name__)

camera        = None
camera_active = False
camera_lock   = threading.Lock()
latest_cam_result = {'emotion': None, 'probs': []}


def preprocess_image_bytes(file):
    try:
        img_data = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (48, 48))
        img = img.astype('float32') / 255.0
        return np.reshape(img, (1, 48, 48, 1))
    except:
        return None


def predict_array(gray_face):
    img = cv2.resize(gray_face, (48, 48)).astype('float32') / 255.0
    img = np.reshape(img, (1, 48, 48, 1))
    return model.predict(img, verbose=0)[0]


def gen_frames():
    global camera, camera_active, latest_cam_result
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    while camera_active:
        with camera_lock:
            if camera is None or not camera.isOpened():
                break
            ok, frame = camera.read()
        if not ok:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=7, minSize=(80, 80)
        )

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
            face_roi = gray[y:y+h, x:x+w]
            pred     = predict_array(face_roi) if model else np.zeros(5)
            top_idx  = int(np.argmax(pred))
            top_em   = EMOTIONS[top_idx]
            top_conf = pred[top_idx]
            meta     = EMOTION_META[top_em]

            latest_cam_result = {
                'emotion': top_em,
                'vn':      meta['vn'],
                'color':   meta['color'],
                'msg':     meta['msg'],
                'conf':    round(float(top_conf) * 100, 1),
                'probs': [
                    {'emotion': EMOTIONS[i],
                     'vn':      EMOTION_META[EMOTIONS[i]]['vn'],
                     'color':   EMOTION_META[EMOTIONS[i]]['color'],
                     'prob':    round(float(pred[i]) * 100, 1)}
                    for i in range(5)
                ]
            }

            # Chỉ vẽ khung mặt, KHÔNG vẽ chữ trong frame
            hex_c = meta['color'].lstrip('#')
            bgr   = tuple(int(hex_c[i:i+2], 16) for i in (4, 2, 0))
            cv2.rectangle(frame, (x, y), (x+w, y+h), bgr, 2)

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')


@app.route('/')
def index():
    return HOME_HTML


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    img = preprocess_image_bytes(file)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    pred    = model.predict(img, verbose=0)[0]
    top_idx = int(np.argmax(pred))
    top_em  = EMOTIONS[top_idx]

    results = sorted(
        [{'emotion': EMOTIONS[i],
          'vn':      EMOTION_META[EMOTIONS[i]]['vn'],
          'color':   EMOTION_META[EMOTIONS[i]]['color'],
          'prob':    round(float(pred[i]) * 100, 2)}
         for i in range(5)],
        key=lambda x: x['prob'], reverse=True
    )
    return jsonify({'results': results, 'top_key': top_em})


@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera, camera_active
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                return jsonify({'ok': False, 'error': 'Cannot open webcam'}), 500
        camera_active = True
    return jsonify({'ok': True})


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera, camera_active
    camera_active = False
    with camera_lock:
        if camera:
            camera.release()
            camera = None
    return jsonify({'ok': True})


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/cam_result')
def cam_result():
    return jsonify(latest_cam_result)


@app.route('/quit', methods=['POST'])
def quit_app():
    global camera, camera_active
    camera_active = False
    with camera_lock:
        if camera:
            camera.release()

    def _stop():
        import time; time.sleep(0.8)
        os._exit(0)
    threading.Thread(target=_stop, daemon=True).start()
    return jsonify({'ok': True})


HOME_HTML = r'''<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nhận dạng cảm xúc khuôn mặt</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700&family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#f7f6f3;
  --surface:#ffffff;
  --surface2:#f0ede8;
  --surface3:#e8e4de;
  --border:#ddd9d2;
  --border2:#c8c4bc;
  --text:#18181a;
  --text2:#6b6760;
  --text3:#a09c96;
  --accent:#2563eb;
  --accent-bg:#eff6ff;
  --green:#16a34a;
  --green-bg:#f0fdf4;
  --red:#dc2626;
  --red-bg:#fef2f2;
  --r:14px;
  --r2:9px;
  --r3:6px;
  --shadow:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.06);
  --shadow-md:0 4px 12px rgba(0,0,0,.10);
}
html,body{height:100%;background:var(--bg);color:var(--text);
  font-family:'Be Vietnam Pro',sans-serif;font-size:14px;line-height:1.6;overflow-x:hidden}

/* ── HEADER ── */
header{
  background:var(--surface);border-bottom:1px solid var(--border);
  padding:0 28px;height:58px;display:flex;align-items:center;gap:12px;
  position:sticky;top:0;z-index:100;box-shadow:var(--shadow);
}
.logo-box{
  width:34px;height:34px;border-radius:9px;background:var(--text);
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
}
.logo-box svg{width:18px;height:18px;fill:none;stroke:#fff;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.logo-name{font-family:'Fraunces',serif;font-size:16px;font-weight:600;letter-spacing:-.2px}
.logo-name em{font-style:italic;color:var(--text2);font-weight:400}
.hdr-right{margin-left:auto;display:flex;align-items:center;gap:8px}
.status-tag{
  display:inline-flex;align-items:center;gap:6px;padding:4px 11px;
  border-radius:99px;font-size:11.5px;font-weight:500;
  background:var(--surface2);border:1px solid var(--border);color:var(--text2);
}
.status-tag.live{background:#fee2e2;border-color:#fca5a5;color:#b91c1c}
.s-dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
.status-tag.live .s-dot{background:#ef4444;animation:pulse .9s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}

/* ── LAYOUT ── */
.layout{display:grid;grid-template-columns:320px 1fr;min-height:calc(100vh - 58px)}
.sidebar{
  background:var(--surface);border-right:1px solid var(--border);
  padding:20px 18px;display:flex;flex-direction:column;gap:18px;overflow-y:auto;
}
.content{padding:22px;display:flex;flex-direction:column;gap:14px;overflow-y:auto}

/* ── SECTION TITLE ── */
.stitle{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:var(--text3);margin-bottom:9px}

/* ── TABS ── */
.tabs{display:flex;border:1px solid var(--border);border-radius:var(--r2);
  overflow:hidden;margin-bottom:14px;background:var(--surface2)}
.tab-btn{flex:1;padding:8px 0;font-size:12.5px;font-weight:500;background:none;
  border:none;cursor:pointer;color:var(--text2);font-family:'Be Vietnam Pro',sans-serif;
  transition:all .15s;border-right:1px solid var(--border)}
.tab-btn:last-child{border-right:none}
.tab-btn.on{background:var(--surface);color:var(--text);font-weight:600;box-shadow:var(--shadow)}
.tab-btn:not(.on):hover{background:var(--surface3)}
.pnl{display:none}.pnl.on{display:block}

/* ── DROP ZONE (unified) ── */
.upload-area{
  border:1.5px dashed var(--border2);border-radius:var(--r2);
  background:var(--surface2);cursor:pointer;transition:all .15s;
  display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:8px;padding:28px 16px;text-align:center;
  position:relative;
}
.upload-area:hover,.upload-area.over{border-color:var(--accent);background:var(--accent-bg)}
.upload-icon{
  width:40px;height:40px;border-radius:10px;background:var(--surface);
  border:1px solid var(--border);display:flex;align-items:center;
  justify-content:center;box-shadow:var(--shadow);
}
.upload-icon svg{width:18px;height:18px;stroke:var(--text2);fill:none;stroke-width:1.8;stroke-linecap:round}
.upload-title{font-size:13px;font-weight:600;color:var(--text)}
.upload-sub{font-size:12px;color:var(--text3)}
#preview-img{
  width:100%;max-height:170px;object-fit:contain;border-radius:var(--r2);
  border:1px solid var(--border);background:var(--surface2);display:none;margin-top:10px;
}
#preview-img.on{display:block}
#file-in{display:none}

/* ── ACTION BUTTON ── */
.detect-btn{
  width:100%;margin-top:10px;padding:11px 0;border-radius:var(--r2);
  font-size:13.5px;font-weight:600;cursor:pointer;
  border:none;background:var(--text);color:#fff;
  font-family:'Be Vietnam Pro',sans-serif;transition:all .15s;
  box-shadow:var(--shadow);
}
.detect-btn:hover{background:#2d2c2a;transform:translateY(-1px);box-shadow:var(--shadow-md)}
.detect-btn:disabled{background:var(--border2);cursor:not-allowed;transform:none;box-shadow:none}

/* ── NOTICE ── */
.notice{padding:9px 11px;border-radius:var(--r3);font-size:12px;
  display:flex;align-items:flex-start;gap:7px;line-height:1.5;margin-top:10px}
.notice.info{background:var(--accent-bg);border:1px solid #bfdbfe;color:#1d4ed8}
.notice.ok  {background:var(--green-bg);border:1px solid #bbf7d0;color:#15803d}
.notice.warn{background:#fffbeb;border:1px solid #fde68a;color:#b45309}
.notice.err {background:var(--red-bg);border:1px solid #fecaca;color:var(--red)}

/* ── EMOTION LEGEND ── */
.em-table{width:100%;border-collapse:collapse}
.em-table tr{border-bottom:1px solid var(--surface2)}
.em-table tr:last-child{border-bottom:none}
.em-table td{padding:7px 8px;font-size:13px}
.em-dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:8px;flex-shrink:0}
.em-en{font-weight:500}
.em-vi{color:var(--text2);font-size:12px;text-align:right}

/* ── CAM CONTROLS ── */
.cam-info{font-size:12.5px;color:var(--text2);line-height:1.6;margin-bottom:12px;
  background:var(--surface2);border-radius:var(--r3);padding:10px 11px;
  border:1px solid var(--border)}
.cam-btns{display:flex;gap:8px}
.cam-btn{flex:1;padding:10px 0;border-radius:var(--r2);font-size:13px;font-weight:600;
  cursor:pointer;border:1px solid var(--border2);background:var(--surface);
  color:var(--text);font-family:'Be Vietnam Pro',sans-serif;transition:all .15s}
.cam-btn:hover{background:var(--surface2)}
.cam-btn.start{background:var(--text);color:#fff;border-color:var(--text)}
.cam-btn.start:hover{background:#2d2c2a}
.cam-btn.stop{background:var(--red-bg);color:var(--red);border-color:#fca5a5}
.cam-btn.stop:hover{background:#fee2e2}

/* ── QUIT ── */
.quit-btn{
  width:100%;padding:9px 0;border-radius:var(--r2);font-size:13px;font-weight:500;
  cursor:pointer;border:1px solid var(--border);background:var(--surface2);
  color:var(--text2);font-family:'Be Vietnam Pro',sans-serif;transition:all .15s;
}
.quit-btn:hover{background:var(--red-bg);border-color:#fca5a5;color:var(--red)}

/* ── IDLE ── */
.idle{display:flex;flex-direction:column;align-items:center;justify-content:center;
  min-height:420px;gap:10px;color:var(--text3);text-align:center}
.idle-circle{width:76px;height:76px;border-radius:50%;background:var(--surface);
  border:1.5px dashed var(--border2);display:flex;align-items:center;
  justify-content:center;opacity:.5}
.idle-circle svg{width:32px;height:32px;stroke:var(--text3);fill:none;
  stroke-width:1.5;stroke-linecap:round}
.idle p{font-size:13.5px;max-width:250px;line-height:1.6}

/* ── RESULT ── */
.result-grid{display:grid;grid-template-columns:auto 1fr;gap:16px;
  background:var(--surface);border:1px solid var(--border);border-radius:var(--r);
  padding:18px;box-shadow:var(--shadow)}
.result-img-wrap{width:180px;flex-shrink:0}
#result-photo{width:180px;height:180px;object-fit:cover;border-radius:var(--r2);
  border:1px solid var(--border);background:var(--surface2);display:none}
#result-photo.on{display:block}
.result-info{display:flex;flex-direction:column;justify-content:space-between;gap:10px}
.result-top-label{
  display:inline-flex;align-items:center;gap:8px;
  padding:6px 14px;border-radius:99px;font-size:13px;font-weight:600;
  border:1.5px solid currentColor;width:fit-content;
}
.result-top-label .dot-big{
  width:9px;height:9px;border-radius:50%;background:currentColor;flex-shrink:0;
}
.result-pct-big{font-family:'Fraunces',serif;font-size:40px;font-weight:700;line-height:1}
.result-vn-label{font-size:13px;color:var(--text2);margin-top:2px}

/* ── BARS ── */
.bars-card{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:16px 18px;box-shadow:var(--shadow)}
.bar-row{display:flex;align-items:center;gap:10px;padding:5px 0;
  border-bottom:1px solid var(--surface2)}
.bar-row:last-child{border-bottom:none}
.bar-label{width:86px;font-size:12.5px;color:var(--text2)}
.bar-track{flex:1;height:5px;background:var(--surface2);border-radius:3px;overflow:hidden}
.bar-fill{height:100%;border-radius:3px;transition:width .75s cubic-bezier(.4,0,.2,1);width:0}
.bar-pct{width:44px;text-align:right;font-size:12.5px;font-weight:600}

/* ── FEEDBACK ── */
.fb-card{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:16px 18px;box-shadow:var(--shadow)}
.fb-q{font-size:13.5px;font-weight:600;margin-bottom:11px}
.fb-row{display:flex;gap:8px}
.fb-btn{padding:7px 16px;border-radius:var(--r3);font-size:13px;font-weight:500;
  cursor:pointer;border:1px solid var(--border2);background:var(--surface2);
  color:var(--text);font-family:'Be Vietnam Pro',sans-serif;transition:all .14s}
.fb-btn:hover{background:var(--surface3)}
.fb-btn.yes{background:var(--green-bg);border-color:#86efac;color:var(--green)}
.fb-btn.yes:hover{background:#dcfce7}
.fb-btn.no{background:var(--red-bg);border-color:#fca5a5;color:var(--red)}
.fb-btn.no:hover{background:#fee2e2}
.fb-ok{padding:9px 12px;border-radius:var(--r3);background:var(--green-bg);
  border:1px solid #86efac;font-size:12.5px;color:var(--green);font-weight:500;display:none}
.fb-ok.on{display:flex;align-items:center;gap:6px}
.wrong-hint{font-size:12.5px;color:var(--text2);margin:10px 0 8px}
.chip-row{display:flex;gap:6px;flex-wrap:wrap}
.chip{padding:5px 12px;border-radius:var(--r3);font-size:12.5px;font-weight:500;
  cursor:pointer;border:1px solid var(--border2);background:var(--surface2);
  color:var(--text);font-family:'Be Vietnam Pro',sans-serif;transition:all .14s}
.chip:hover{border-color:var(--text)}
.chip.picked{background:var(--text);color:#fff;border-color:var(--text)}
.fb-thanks{margin-top:10px;padding:9px 12px;border-radius:var(--r3);
  background:var(--green-bg);border:1px solid #86efac;font-size:12.5px;
  color:var(--green);display:none}
.fb-thanks.on{display:block}

/* ── WEBCAM ── */
.cam-panel{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);overflow:hidden;box-shadow:var(--shadow);display:none}
.cam-panel.on{display:block}
.cam-video-wrap{position:relative;background:#000;line-height:0}
#cam-stream{width:100%;display:block;max-height:420px;object-fit:contain}
.live-badge{position:absolute;top:10px;left:10px;
  background:rgba(220,38,38,.88);color:#fff;
  font-size:11px;font-weight:700;letter-spacing:1.4px;
  padding:3px 10px;border-radius:99px;display:flex;align-items:center;gap:5px}
.live-dot{width:5px;height:5px;border-radius:50%;background:#fff;animation:pulse .9s infinite}

/* Kết quả webcam bên dưới video */
.cam-data{padding:14px 18px;border-top:1px solid var(--border)}
.cam-top-row{display:flex;align-items:center;justify-content:space-between;
  margin-bottom:12px;gap:12px}
.cam-emotion-tag{display:flex;align-items:center;gap:8px}
.cam-dot-big{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.cam-emotion-name{font-family:'Fraunces',serif;font-size:18px;font-weight:600}
.cam-conf-pct{font-family:'Fraunces',serif;font-size:18px;font-weight:600;color:var(--text2)}
.cam-bars{display:flex;flex-direction:column;gap:5px}
.cam-bar-row{display:flex;align-items:center;gap:9px}
.cam-bar-lbl{width:80px;font-size:12px;color:var(--text2)}
.cam-bar-track{flex:1;height:4px;background:var(--surface2);border-radius:2px;overflow:hidden}
.cam-bar-fill{height:100%;border-radius:2px;transition:width .35s ease}
.cam-bar-pct{width:38px;text-align:right;font-size:12px;font-weight:600}

/* Thông điệp webcam */
.cam-msg{padding:11px 18px;font-size:13px;line-height:1.55;
  display:none;border-top:1px solid var(--border);
  display:flex;gap:8px;align-items:flex-start}
.cam-msg.on{display:flex}
.msg-bar{width:3px;border-radius:2px;flex-shrink:0;align-self:stretch}

@media(max-width:820px){
  .layout{grid-template-columns:1fr}
  .sidebar{border-right:none;border-bottom:1px solid var(--border)}
  .result-grid{grid-template-columns:1fr}
  #result-photo{width:100%;height:200px}
  .result-img-wrap{width:100%}
}
</style>
</head>
<body>

<header>
  <div class="logo-box">
    <svg viewBox="0 0 24 24"><circle cx="12" cy="9" r="3.5"/>
      <path d="M5 20c0-3.9 3.1-7 7-7s7 3.1 7 7"/>
      <path d="M8.5 9.5c.3-.8 1-1.5 2-1.8M15.5 11a3.5 3.5 0 01-7 0"/>
    </svg>
  </div>
  <div class="logo-name">Nhận dạng <em>cảm xúc khuôn mặt</em></div>
  <div class="hdr-right">
    <div class="status-tag" id="status-tag">
      <div class="s-dot"></div><span>Model sẵn sàng</span>
    </div>
  </div>
</header>

<div class="layout">

  <!-- ══ SIDEBAR ══ -->
  <div class="sidebar">

    <div>
      <div class="tabs">
        <button class="tab-btn on" onclick="switchTab('upload',this)">Nhận dạng ảnh</button>
        <button class="tab-btn"    onclick="switchTab('camera',this)">Webcam trực tiếp</button>
      </div>

      <!-- TAB: Upload -->
      <div id="tab-upload" class="pnl on">
        <div class="stitle">Tải ảnh lên</div>

        <!-- Vùng kéo thả + xem trước — GỘP LÀM MỘT -->
        <div class="upload-area" id="drop-zone"
          onclick="document.getElementById('file-in').click()"
          ondragover="event.preventDefault();this.classList.add('over')"
          ondragleave="this.classList.remove('over')"
          ondrop="handleDrop(event)">
          <div class="upload-icon">
            <svg viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
          </div>
          <div class="upload-title" id="drop-title">Nhấn hoặc kéo thả ảnh vào đây</div>
          <div class="upload-sub" id="drop-sub">Hỗ trợ JPG, PNG, WEBP</div>
          <!-- preview nằm ngay trong vùng drop -->
          <img id="preview-img" alt="preview">
        </div>
        <input type="file" id="file-in" accept="image/*" onchange="handleFile(event)">

        <button class="detect-btn" id="btn-detect" onclick="runDetect()">Nhận dạng cảm xúc</button>

        <div id="notice-wrap">
          <div class="notice info">
            Tải lên ảnh khuôn mặt rõ nét rồi nhấn <strong>Nhận dạng cảm xúc</strong>.
          </div>
        </div>
      </div>

      <!-- TAB: Camera -->
      <div id="tab-camera" class="pnl">
        <div class="stitle">Webcam</div>
        <div class="cam-info">
          Model phân tích cảm xúc theo thời gian thực. Hãy đảm bảo khuôn mặt được chiếu sáng đủ và nhìn thẳng vào camera.
        </div>
        <div class="cam-btns">
          <button class="cam-btn start" id="btn-start-cam" onclick="startCam()">Bật camera</button>
          <button class="cam-btn stop"  id="btn-stop-cam"  onclick="stopCam()" style="display:none">Tắt camera</button>
        </div>
      </div>
    </div>

    <!-- Bảng nhãn -->
    <div>
      <div class="stitle">5 cảm xúc nhận dạng</div>
      <table class="em-table">
        <tr><td><span class="em-dot" style="background:#ef4444"></span><span class="em-en">Angry</span></td><td class="em-vi">Tức giận</td></tr>
        <tr><td><span class="em-dot" style="background:#f59e0b"></span><span class="em-en">Happy</span></td><td class="em-vi">Vui vẻ</td></tr>
        <tr><td><span class="em-dot" style="background:#64748b"></span><span class="em-en">Neutral</span></td><td class="em-vi">Bình thường</td></tr>
        <tr><td><span class="em-dot" style="background:#3b82f6"></span><span class="em-en">Sad</span></td><td class="em-vi">Buồn bã</td></tr>
        <tr><td><span class="em-dot" style="background:#a855f7"></span><span class="em-en">Surprise</span></td><td class="em-vi">Ngạc nhiên</td></tr>
      </table>
    </div>

    <div style="margin-top:auto">
      <button class="quit-btn" onclick="doQuit()">Tắt ứng dụng</button>
    </div>
  </div>

  <!-- ══ CONTENT ══ -->
  <div class="content">

    <!-- Idle -->
    <div class="idle" id="idle">
      <div class="idle-circle">
        <svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/>
          <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
          <circle cx="8.5" cy="7.5" r=".5" fill="currentColor"/>
          <circle cx="15.5" cy="7.5" r=".5" fill="currentColor"/>
        </svg>
      </div>
      <p>Tải ảnh lên rồi nhấn <strong>Nhận dạng cảm xúc</strong>,<br>hoặc bật <strong>Webcam trực tiếp</strong>.</p>
    </div>

    <!-- Kết quả ảnh -->
    <div id="result-wrap" style="display:none;flex-direction:column;gap:14px">

      <div class="result-grid">
        <div class="result-img-wrap">
          <img id="result-photo" alt="result">
        </div>
        <div class="result-info">
          <div>
            <div class="stitle">Kết quả nhận dạng</div>
            <div class="result-top-label" id="r-label">
              <div class="dot-big"></div><span id="r-name-en"></span>
            </div>
            <div style="margin-top:10px">
              <div class="result-pct-big" id="r-pct"></div>
              <div class="result-vn-label" id="r-vn"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Bars xác suất -->
      <div class="bars-card">
        <div class="stitle" style="margin-bottom:6px">Phân bố xác suất</div>
        <div id="r-bars"></div>
      </div>

      <!-- Feedback -->
      <div class="fb-card">
        <div class="fb-q">Kết quả có chính xác không?</div>
        <div id="fb1" class="fb-row">
          <button class="fb-btn yes" onclick="fbYes()">Đúng rồi</button>
          <button class="fb-btn no"  onclick="fbNo()">Chưa đúng</button>
        </div>
        <div class="fb-ok" id="fb-ok">✓ Cảm ơn bạn đã xác nhận!</div>
        <div id="fb-wrong-wrap" style="display:none">
          <div class="wrong-hint">Cảm xúc thực tế trong ảnh là gì?</div>
          <div class="chip-row" id="chip-row"></div>
          <div class="fb-thanks" id="fb-thanks"></div>
        </div>
      </div>
    </div>

    <!-- Webcam panel -->
    <div class="cam-panel" id="cam-panel">
      <div class="cam-video-wrap">
        <img id="cam-stream" src="" alt="webcam">
        <div class="live-badge"><div class="live-dot"></div> LIVE</div>
      </div>

      <div class="cam-data">
        <div class="cam-top-row">
          <div class="cam-emotion-tag">
            <div class="cam-dot-big" id="c-dot"></div>
            <div class="cam-emotion-name" id="c-name">—</div>
          </div>
          <div class="cam-conf-pct" id="c-conf"></div>
        </div>
        <div class="cam-bars" id="c-bars"></div>
      </div>

      <div class="cam-msg" id="c-msg">
        <div class="msg-bar" id="c-bar"></div>
        <span id="c-msg-txt"></span>
      </div>
    </div>

  </div>
</div>

<script>
const $ = id => document.getElementById(id)
let curFile = null, curTopKey = null, camOn = false
let camTimer = null, lastEmotion = null

const META = {
  angry:   {vn:'Tức giận',   color:'#ef4444'},
  happy:   {vn:'Vui vẻ',     color:'#f59e0b'},
  neutral: {vn:'Bình thường',color:'#64748b'},
  sad:     {vn:'Buồn bã',    color:'#3b82f6'},
  surprise:{vn:'Ngạc nhiên', color:'#a855f7'},
}

// ── TABS ──────────────────────────────────────────────────────
function switchTab(tab, btn) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('on'))
  btn.classList.add('on')
  $('tab-upload').classList.remove('on')
  $('tab-camera').classList.remove('on')
  $('tab-' + tab).classList.add('on')
  if (tab === 'camera') {
    $('idle').style.display = 'none'
    $('result-wrap').style.display = 'none'
  } else {
    stopCam()
    $('cam-panel').classList.remove('on')
    if (!curTopKey) $('idle').style.display = 'flex'
  }
}

// ── FILE ──────────────────────────────────────────────────────
function handleDrop(e) {
  e.preventDefault()
  $('drop-zone').classList.remove('over')
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0])
}
function handleFile(e) { if (e.target.files[0]) loadFile(e.target.files[0]) }

function loadFile(file) {
  curFile = file
  const reader = new FileReader()
  reader.onload = ev => {
    const img = $('preview-img')
    img.src = ev.target.result
    img.classList.add('on')
    $('drop-title').textContent = file.name
    $('drop-sub').textContent   = (file.size/1024).toFixed(0) + ' KB — nhấn để đổi ảnh'
  }
  reader.readAsDataURL(file)
  resetResult()
  setNotice('Ảnh đã chọn. Nhấn <strong>Nhận dạng cảm xúc</strong> để tiếp tục.', 'ok')
}

function setNotice(msg, type) {
  $('notice-wrap').innerHTML =
    `<div class="notice ${type}">${msg}</div>`
}

// ── DETECT ────────────────────────────────────────────────────
function runDetect() {
  if (!curFile) { setNotice('Hãy chọn ảnh trước.', 'warn'); return }
  $('btn-detect').disabled = true
  $('btn-detect').textContent = 'Đang phân tích...'
  setNotice('Đang xử lý...', 'info')

  const fd = new FormData()
  fd.append('file', curFile)
  fetch('/predict', { method:'POST', body:fd })
    .then(r => r.json())
    .then(d => {
      if (d.error) { setNotice('Lỗi: ' + d.error, 'err'); return }
      showResult(d)
      setNotice('Nhận dạng hoàn tất.', 'ok')
    })
    .catch(() => setNotice('Lỗi kết nối server.', 'err'))
    .finally(() => {
      $('btn-detect').disabled = false
      $('btn-detect').textContent = 'Nhận dạng cảm xúc'
    })
}

function showResult(d) {
  const top = d.results[0]
  curTopKey = d.top_key

  $('idle').style.display = 'none'
  $('result-wrap').style.display = 'flex'

  // Ảnh
  const rp = $('result-photo')
  rp.src = $('preview-img').src
  rp.classList.add('on')

  // Label & %
  const lbl = $('r-label')
  lbl.style.color = top.color
  lbl.querySelector('.dot-big').style.background = top.color
  $('r-name-en').textContent = top.emotion.charAt(0).toUpperCase() + top.emotion.slice(1)
  $('r-pct').textContent = top.prob.toFixed(1) + '%'
  $('r-pct').style.color = top.color
  $('r-vn').textContent  = top.vn

  // Bars
  const bars = $('r-bars')
  bars.innerHTML = ''
  d.results.forEach(r => {
    bars.innerHTML += `
      <div class="bar-row">
        <div class="bar-label" style="color:${r.prob === top.prob ? r.color : ''};
          font-weight:${r.prob === top.prob ? '600' : '400'}">${r.vn}</div>
        <div class="bar-track">
          <div class="bar-fill" data-w="${r.prob}" style="background:${r.color}"></div>
        </div>
        <div class="bar-pct" style="color:${r.color}">${r.prob.toFixed(1)}%</div>
      </div>`
  })
  requestAnimationFrame(() => requestAnimationFrame(() => {
    document.querySelectorAll('#r-bars .bar-fill').forEach(b => {
      b.style.width = b.dataset.w + '%'
    })
  }))

  resetFeedback()
}

function resetResult() {
  $('result-wrap').style.display = 'none'
  $('result-photo').classList.remove('on')
  curTopKey = null
  if (!camOn) $('idle').style.display = 'flex'
}

// ── FEEDBACK ──────────────────────────────────────────────────
function resetFeedback() {
  $('fb1').style.display = 'flex'
  $('fb-ok').classList.remove('on')
  $('fb-wrong-wrap').style.display = 'none'
  $('fb-thanks').classList.remove('on')
  $('chip-row').innerHTML = ''
}
function fbYes() {
  $('fb1').style.display = 'none'
  $('fb-ok').classList.add('on')
}
function fbNo() {
  $('fb1').style.display = 'none'
  $('fb-wrong-wrap').style.display = 'block'
  const row = $('chip-row')
  row.innerHTML = ''
  Object.entries(META).forEach(([k, m]) => {
    const btn = document.createElement('button')
    btn.className = 'chip' + (k === curTopKey ? ' picked' : '')
    btn.textContent = m.vn
    btn.onclick = () => {
      document.querySelectorAll('.chip').forEach(c => c.classList.remove('picked'))
      btn.classList.add('picked')
      const t = $('fb-thanks')
      t.innerHTML = `✓ Cảm ơn! Nhãn đúng là <strong>${m.vn}</strong>. Phản hồi của bạn giúp cải thiện model.`
      t.classList.add('on')
    }
    row.appendChild(btn)
  })
}

// ── WEBCAM ────────────────────────────────────────────────────
function startCam() {
  fetch('/start_camera', {method:'POST'})
    .then(r => r.json())
    .then(d => {
      if (!d.ok) { alert('Lỗi mở webcam: ' + d.error); return }
      camOn = true
      $('idle').style.display = 'none'
      $('result-wrap').style.display = 'none'
      $('cam-panel').classList.add('on')
      $('cam-stream').src = '/video_feed'
      $('btn-start-cam').style.display = 'none'
      $('btn-stop-cam').style.display  = ''
      const st = $('status-tag')
      st.className = 'status-tag live'
      st.innerHTML = '<div class="s-dot"></div><span>Đang phát</span>'
      camTimer = setInterval(pollCam, 550)
    })
}

function stopCam() {
  if (!camOn) return
  camOn = false
  clearInterval(camTimer)
  fetch('/stop_camera', {method:'POST'})
  $('cam-stream').src = ''
  $('cam-panel').classList.remove('on')
  $('btn-start-cam').style.display = ''
  $('btn-stop-cam').style.display  = 'none'
  const st = $('status-tag')
  st.className = 'status-tag'
  st.innerHTML = '<div class="s-dot"></div><span>Model sẵn sàng</span>'
  lastEmotion = null
  if (!curTopKey) $('idle').style.display = 'flex'
}

function pollCam() {
  fetch('/cam_result').then(r => r.json()).then(d => {
    if (!d.emotion) return
    // Tên & màu
    $('c-dot').style.background = d.color
    $('c-name').textContent = d.vn
    $('c-name').style.color = d.color
    $('c-conf').textContent = d.conf + '%'

    // Bars
    const cb = $('c-bars')
    cb.innerHTML = ''
    d.probs.forEach(p => {
      cb.innerHTML += `
        <div class="cam-bar-row">
          <div class="cam-bar-lbl">${p.vn}</div>
          <div class="cam-bar-track">
            <div class="cam-bar-fill" style="width:${p.prob}%;background:${p.color}"></div>
          </div>
          <div class="cam-bar-pct" style="color:${p.color}">${p.prob}%</div>
        </div>`
    })

    // Thông điệp — đổi khi cảm xúc thay đổi
    if (d.emotion !== lastEmotion) {
      lastEmotion = d.emotion
      const msgColors = {
        angry: {bg:'#fff7ed',bar:'#f97316',text:'#9a3412'},
        happy: {bg:'#fefce8',bar:'#f59e0b',text:'#78350f'},
        neutral:{bg:'#f8fafc',bar:'#64748b',text:'#334155'},
        sad:   {bg:'#eff6ff',bar:'#3b82f6',text:'#1e3a8a'},
        surprise:{bg:'#faf5ff',bar:'#a855f7',text:'#581c87'},
      }
      const mc = msgColors[d.emotion] || msgColors.neutral
      const box = $('c-msg')
      box.style.background = mc.bg
      box.style.borderTopColor = mc.bar + '44'
      $('c-bar').style.background = mc.bar
      $('c-msg-txt').style.color  = mc.text
      $('c-msg-txt').textContent  = d.msg
      box.classList.add('on')
    }
  })
}

// ── QUIT ──────────────────────────────────────────────────────
function doQuit() {
  if (!confirm('Tắt ứng dụng?')) return
  clearInterval(camTimer)
  fetch('/quit', {method:'POST'}).finally(() => {
    document.body.innerHTML = `
      <div style="display:flex;height:100vh;align-items:center;justify-content:center;
        flex-direction:column;gap:14px;background:#f7f6f3;font-family:sans-serif">
        <div style="font-size:44px">👋</div>
        <p style="color:#6b6760">Ứng dụng đã tắt. Bạn có thể đóng tab này.</p>
      </div>`
  })
}
</script>
</body>
</html>
'''

if __name__ == '__main__':
    print("🌐  http://127.0.0.1:5000")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)