#!/usr/bin/env python3
"""
KV260 Pharmaceutical Visual Inspection Dashboard (Flask)
Includes live KV260 hardware telemetry (DRAM, CPU %, SOM Power) via /hw_stats.
"""

import os
import glob
import json
import subprocess
import threading
from flask import Flask, render_template_string, send_file, redirect, url_for, jsonify

RESULTS_DIR  = "/home/root/kv260_test/results"
PIPELINE_PY  = "/home/root/kv260_test/usb_dpu_llama.py"
METRICS_FILE = f"{RESULTS_DIR}/latest_metrics.json"
FRAME_FILE   = f"{RESULTS_DIR}/latest_frame.jpg"
VIDEO_FILE   = f"{RESULTS_DIR}/usb_dpu_output.avi"

app = Flask(__name__)
run_lock = threading.Lock()
_prev_cpu = None


def read_hw_stats():
    global _prev_cpu

    # 1. DRAM (RAM) from /proc/meminfo
    mem_total_kb, mem_avail_kb = 0, 0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_avail_kb = int(line.split()[1])
    except Exception:
        pass
    mem_total_mb = round(mem_total_kb / 1024)
    mem_used_mb  = round((mem_total_kb - mem_avail_kb) / 1024)
    mem_pct      = round((mem_used_mb / mem_total_mb) * 100, 1) if mem_total_mb else 0.0

    # 2. CPU % from /proc/stat delta
    cpu_pct = 0.0
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            parts = list(map(int, f.readline().split()[1:8]))
        idle = parts[3] + parts[4]
        total = sum(parts)
        if _prev_cpu is not None:
            d_total = total - _prev_cpu[0]
            d_idle  = idle - _prev_cpu[1]
            if d_total > 0:
                cpu_pct = round((1.0 - d_idle / d_total) * 100.0, 1)
        _prev_cpu = (total, idle)
    except Exception:
        pass

    # 3. SOM Board Power (W) from INA260 hwmon sysfs
    power_w = "N/A"
    for pfile in glob.glob("/sys/class/hwmon/hwmon*/power1_input"):
        try:
            with open(pfile, "r", encoding="utf-8") as f:
                uw = int(f.read().strip())
            power_w = f"{uw / 1_000_000:.2f} W"
            break
        except Exception:
            continue

    return {
        "ram_used_mb": mem_used_mb,
        "ram_total_mb": mem_total_mb,
        "ram_pct": mem_pct,
        "cpu_pct": cpu_pct,
        "power_w": power_w,
        "state": "RUNNING (DPU / LLAMA)" if run_lock.locked() else "IDLE (READY)",
    }


HTML = """
<!doctype html>
<html>
<head>
  <title>KV260 Pharma QC — YOLOv4 + ExecuTorch Llama 3.2</title>
  {% if is_running %}<meta http-equiv="refresh" content="5">{% endif %}
  <style>
    body { font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }
    h1 { margin: 0 0 6px; color: #38bdf8; }
    .sub { color: #94a3b8; margin-bottom: 18px; font-size: 14px; }
    .btn { background: #2563eb; color: #fff; border: none; padding: 12px 20px; border-radius: 6px;
           font-size: 15px; cursor: pointer; text-decoration: none; display: inline-block; margin-right: 10px; }
    .btn:hover { background: #1d4ed8; }
    .btn:disabled { background: #475569; cursor: not-allowed; }
    .btn-sec { background: #334155; }
    .banner { background: #1e3a8a; color: #bfdbfe; padding: 12px 16px; border-radius: 8px;
              margin-top: 16px; border: 1px solid #3b82f6; font-weight: bold; }
    .hw-bar { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 18px; }
    .hw-item { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px 14px; }
    .hw-item .lbl { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
    .hw-item .val { font-size: 19px; font-weight: bold; margin-top: 4px; color: #38bdf8; }
    .grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 20px; margin-top: 18px; }
    .card { background: #1e293b; border-radius: 10px; padding: 18px; border: 1px solid #334155; }
    .card h2 { margin-top: 0; font-size: 17px; color: #f8fafc; border-bottom: 1px solid #334155; padding-bottom: 8px; }
    img { width: 100%; border-radius: 6px; border: 1px solid #475569; }
    .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px; }
    .stat { background: #0f172a; padding: 12px; border-radius: 8px; }
    .stat .lbl { font-size: 12px; color: #94a3b8; }
    .stat .val { font-size: 22px; font-weight: bold; margin-top: 4px; }
    .status { padding: 10px 14px; border-radius: 6px; font-weight: bold; text-align: center; margin-bottom: 14px; }
    .ok { background: #14532d; color: #86efac; }
    .warn { background: #7f1d1d; color: #fca5a5; }
    .neutral { background: #334155; color: #cbd5e1; }
    .llama-box { background: #0f172a; padding: 14px; border-radius: 8px; line-height: 1.55; white-space: pre-wrap; }
  </style>
</head>
<body>
  <h1>KV260 Pharmaceutical Visual Inspection</h1>
  <div class="sub">DPU-Accelerated YOLOv4 (512x512 INT8) + On-Device ExecuTorch Llama 3.2 1B (4-bit) — 100% Offline</div>

  <form action="/run" method="post" style="display:inline;">
    <button class="btn" type="submit" {% if is_running %}disabled{% endif %}>
      {{ 'Inspection Running...' if is_running else 'Run Live USB Inspection' }}
    </button>
  </form>
  {% if has_video %}
    <a class="btn btn-sec" href="/video">Download Annotated Video (.avi)</a>
  {% endif %}

  {% if is_running %}
    <div class="banner">Running 10s YOLOv4 DPU capture + ExecuTorch Llama 3.2 1B (~75s total). Page auto-refreshes every 5s.</div>
  {% endif %}

  <div class="hw-bar">
    <div class="hw-item"><div class="lbl">KV260 DRAM Usage</div><div class="val" id="hw-ram">{{ hw.ram_used_mb }} / {{ hw.ram_total_mb }} MB ({{ hw.ram_pct }}%)</div></div>
    <div class="hw-item"><div class="lbl">Arm Cortex-A53 CPU</div><div class="val" id="hw-cpu">{{ hw.cpu_pct }}%</div></div>
    <div class="hw-item"><div class="lbl">SOM Board Power</div><div class="val" id="hw-pwr">{{ hw.power_w }}</div></div>
    <div class="hw-item"><div class="lbl">Pipeline State</div><div class="val" id="hw-state" style="color:#facc15;">{{ hw.state }}</div></div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Latest Annotated Frame</h2>
      {% if has_frame %}
        <img src="/frame?t={{ ts }}" alt="Latest inspection frame">
      {% else %}
        <p>No inspection frame captured yet. Click <b>Run Live USB Inspection</b>.</p>
      {% endif %}
    </div>

    <div>
      <div class="card" style="margin-bottom: 20px;">
        <h2>YOLOv4 DPU Inspection Metrics</h2>
        {% if m %}
          <div class="status {{ 'warn' if 'MANUAL' in m.qc_status else ('ok' if 'NO DEFECT' in m.qc_status else 'neutral') }}">
            {{ m.qc_status }}
          </div>
          <div class="stat-grid">
            <div class="stat"><div class="lbl">Normal Detections</div><div class="val" style="color:#4ade80;">{{ m.normal_count }} ({{ m.normal_avg_conf }}%)</div></div>
            <div class="stat"><div class="lbl">Defective Detections</div><div class="val" style="color:#f87171;">{{ m.defective_count }} ({{ m.defective_avg_conf }}%)</div></div>
            <div class="stat"><div class="lbl">Average DPU FPS</div><div class="val">{{ m.avg_fps }}</div></div>
            <div class="stat"><div class="lbl">Frames Analyzed</div><div class="val">{{ m.total_frames }}</div></div>
          </div>
          <div class="sub" style="margin:0;">Last run: {{ m.timestamp }} ({{ m.duration_sec }}s)</div>
        {% else %}
          <p>No metrics available yet.</p>
        {% endif %}
      </div>

      <div class="card">
        <h2>ExecuTorch Llama 3.2 1B — QC Summary</h2>
        <div class="llama-box">{{ m.llama_summary if m and m.llama_summary else 'Run an inspection to generate the on-device Llama 3.2 QC explanation.' }}</div>
      </div>
    </div>
  </div>

  <script>
    async function pollHW() {
      try {
        const r = await fetch('/hw_stats');
        const d = await r.json();
        document.getElementById('hw-ram').innerText = `${d.ram_used_mb} / ${d.ram_total_mb} MB (${d.ram_pct}%)`;
        document.getElementById('hw-cpu').innerText = `${d.cpu_pct}%`;
        document.getElementById('hw-pwr').innerText = d.power_w;
        document.getElementById('hw-state').innerText = d.state;
      } catch (e) {}
    }
    setInterval(pollHW, 2000);
  </script>
</body>
</html>
"""


def _run_pipeline_bg():
    try:
        subprocess.run(
            ["python3", PIPELINE_PY, "--camera", "0", "--duration", "10", "--conf", "0.45"],
            check=False,
        )
    finally:
        run_lock.release()


@app.route("/")
def index():
    metrics = None
    if os.path.isfile(METRICS_FILE):
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    ts = os.path.getmtime(FRAME_FILE) if os.path.isfile(FRAME_FILE) else 0
    return render_template_string(
        HTML,
        m=metrics,
        hw=read_hw_stats(),
        has_frame=os.path.isfile(FRAME_FILE),
        has_video=os.path.isfile(VIDEO_FILE),
        is_running=run_lock.locked(),
        ts=ts,
    )


@app.route("/hw_stats")
def hw_stats():
    return jsonify(read_hw_stats())


@app.route("/frame")
def frame():
    resp = send_file(FRAME_FILE, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


@app.route("/video")
def video():
    return send_file(VIDEO_FILE, as_attachment=True, download_name="usb_dpu_output.avi")


@app.route("/run", methods=["POST"])
def run_inspection():
    if run_lock.acquire(blocking=False):
        threading.Thread(target=_run_pipeline_bg, daemon=True).start()
    return redirect(url_for("index"))


if __name__ == "__main__":
    for f in (METRICS_FILE, FRAME_FILE, VIDEO_FILE):
        if os.path.isfile(f):
            os.remove(f)
    app.run(host="0.0.0.0", port=5000, debug=False)
