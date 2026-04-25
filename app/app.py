import os
import time
import json
import atexit
import threading
import smtplib
import ssl
from email.message import EmailMessage
from app.routes.chat_routes import chat_bp
from app.services.engine_state_store import EngineStateStore, set_engine_store




import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
app.register_blueprint(chat_bp)

# =====================
# PATHS
# =====================

RF_MODEL_PATH = os.path.join(PROJECT_DIR, "model", "rf_rul_pipeline", "model.pkl")
RF_CONFIG_PATH = os.path.join(PROJECT_DIR, "model", "rf_rul_pipeline", "config.pkl")

GRU_MODEL_PATH = os.path.join(PROJECT_DIR, "model", "gru_rul_pipeline", "model.keras")
GRU_CONFIG_PATH = os.path.join(PROJECT_DIR, "model", "gru_rul_pipeline", "config.pkl")

AE_MODEL_PATH = os.path.join(PROJECT_DIR, "model", "ae_pipeline", "model.keras")
AE_CONFIG_PATH = os.path.join(PROJECT_DIR, "model", "ae_pipeline", "config.pkl")

TEST_DATA_PATH = os.path.join(PROJECT_DIR, "data_csv", "test_FD001.csv")
STATE_PATH = os.path.join(PROJECT_DIR, "engine_runtime_state.json")

print("BASE_DIR:", BASE_DIR)
print("PROJECT_DIR:", PROJECT_DIR)
print("TEST_DATA_PATH:", TEST_DATA_PATH)
print("TEST_DATA_PATH exists:", os.path.exists(TEST_DATA_PATH))
print("RF_MODEL_PATH exists:", os.path.exists(RF_MODEL_PATH))
print("GRU_MODEL_PATH exists:", os.path.exists(GRU_MODEL_PATH))
print("AE_MODEL_PATH exists:", os.path.exists(AE_MODEL_PATH))

# =====================
# LOAD DATA
# =====================

df = pd.read_csv(TEST_DATA_PATH)

sensor_map = {
    "sensor_1": "fan_inlet_temperature_T2",
    "sensor_2": "compressor_LPC_outlet_temp_T24",
    "sensor_3": "compressor_HPC_outlet_temp_T30",
    "sensor_4": "turbine_LPT_outlet_temp_T50",
    "sensor_5": "fan_inlet_pressure_P2",
    "sensor_6": "bypass_duct_pressure_P15",
    "sensor_7": "HPC_outlet_pressure_P30",
    "sensor_8": "fan_speed_Nf",
    "sensor_9": "core_speed_Nc",
    "sensor_10": "engine_pressure_ratio_EPR",
    "sensor_11": "static_pressure_HPC_Ps30",
    "sensor_12": "fuel_flow_ratio_phi",
    "sensor_13": "corrected_fan_speed",
    "sensor_14": "corrected_core_speed",
    "sensor_15": "bypass_ratio_BPR",
    "sensor_16": "fuel_air_ratio_FAR",
    "sensor_17": "bleed_air_enthalpy",
    "sensor_18": "bleed_air_pressure",
    "sensor_19": "HPT_cooling_flow",
    "sensor_20": "LPT_cooling_flow",
    "sensor_21": "exhaust_gas_temperature_EGT"
}

df = df.rename(columns=sensor_map)
df["engine_id"] = df["engine_id"].astype(str)

# =====================
# LOAD MODELS
# =====================

rf_model = joblib.load(RF_MODEL_PATH)
rf_config = joblib.load(RF_CONFIG_PATH)

gru_model = tf.keras.models.load_model(GRU_MODEL_PATH)
gru_config = joblib.load(GRU_CONFIG_PATH)

ae_model = tf.keras.models.load_model(AE_MODEL_PATH)
ae_config = joblib.load(AE_CONFIG_PATH)


# =====================
# FEATURE CONFIG
# =====================

def map_cols(cols):
    return [sensor_map.get(col, col) for col in cols]

rf_scaler = rf_config["scaler"]
rf_feature_cols = map_cols(rf_config["feature_cols"])

gru_scaler = gru_config.get("scaler", rf_scaler)
gru_feature_cols = map_cols(gru_config.get("feature_cols", rf_config["feature_cols"]))

ae_scaler = ae_config.get("scaler", rf_scaler)
ae_feature_cols = map_cols(ae_config.get("feature_cols", rf_config["feature_cols"]))

GRU_SEQ_LEN = int(gru_config["seq_len"])
AE_SEQ_LEN = int(ae_config["seq_len"])

ae_expected_features = None
if isinstance(ae_model.input_shape, tuple) and len(ae_model.input_shape) == 3:
    ae_expected_features = ae_model.input_shape[-1]

if ae_expected_features is not None and len(ae_feature_cols) != ae_expected_features:
    if len(ae_feature_cols) == ae_expected_features + 1 and "unknown_sensor" in ae_feature_cols:
        ae_feature_cols = [c for c in ae_feature_cols if c != "unknown_sensor"]
    else:
        ae_feature_cols = ae_feature_cols[:ae_expected_features]

# =====================
# COLUMN GROUPS
# =====================

base_exclude_cols = {"engine_id", "cycle", "RUL", "rul"}
preferred_op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
op_cols = [col for col in preferred_op_cols if col in df.columns]

if len(op_cols) == 0:
    op_cols = [col for col in df.columns if "op" in col.lower()]

sensor_cols = [
    col for col in df.columns
    if col not in base_exclude_cols and col not in op_cols
]
engine_store = EngineStateStore(
    df=df,
    rf_model=rf_model,
    rf_scaler=rf_scaler,
    rf_feature_cols=rf_feature_cols,
    gru_model=gru_model,
    gru_scaler=gru_scaler,
    gru_feature_cols=gru_feature_cols,
    gru_seq_len=GRU_SEQ_LEN,
    ae_model=ae_model,
    ae_scaler=ae_scaler,
    ae_feature_cols=ae_feature_cols,
    ae_seq_len=AE_SEQ_LEN,
    warning_th=float(ae_config["warning_th"]),
    critical_th=float(ae_config["critical_th"]),
    op_cols=op_cols,
    sensor_cols=sensor_cols,
)

set_engine_store(engine_store)

# =====================
# UTILS
# =====================

def get_engine_data(engine_id):
    return df[df["engine_id"].astype(str) == str(engine_id)].copy()

def prepare_features_for_model(df_engine, feature_cols, scaler, model_name="model"):
    missing_cols = [col for col in feature_cols if col not in df_engine.columns]
    if missing_cols:
        raise ValueError(f"{model_name} missing feature columns: {missing_cols}")

    X = df_engine[feature_cols].copy()
    return scaler.transform(X)

def create_sequence(X, seq_len):
    if len(X) < seq_len:
        return None
    return np.array([X[-seq_len:]], dtype=np.float32)

def predict_rul_from_engine_df(df_engine):
    if len(df_engine) == 0:
        return {"rul": None, "model": None, "message": "No data"}

    X_rf = prepare_features_for_model(df_engine, rf_feature_cols, rf_scaler, model_name="RF")

    if len(X_rf) < GRU_SEQ_LEN:
        pred = rf_model.predict(X_rf[-1].reshape(1, -1))[0]
        return {
            "rul": float(pred),
            "model": "RF",
            "message": f"Chưa đủ {GRU_SEQ_LEN} bước, dùng Random Forest"
        }

    X_gru = prepare_features_for_model(df_engine, gru_feature_cols, gru_scaler, model_name="GRU")

    seq = create_sequence(X_gru, GRU_SEQ_LEN)
    if seq is None:
        pred = rf_model.predict(X_rf[-1].reshape(1, -1))[0]
        return {
            "rul": float(pred),
            "model": "RF",
            "message": "Không tạo được sequence GRU, fallback Random Forest"
        }

    pred = gru_model.predict(seq, verbose=0)[0][0]
    return {
        "rul": float(pred),
        "model": "GRU",
        "message": f"Đủ chuỗi {GRU_SEQ_LEN} bước, dùng GRU"
    }

def predict_anomaly_from_engine_df(df_engine):
    warning_th = float(ae_config["warning_th"])
    critical_th = float(ae_config["critical_th"])

    if len(df_engine) == 0:
        return {
            "score": None,
            "status": "No data",
            "warning_th": warning_th,
            "critical_th": critical_th
        }

    X_ae = prepare_features_for_model(df_engine, ae_feature_cols, ae_scaler, model_name="AE")

    seq = create_sequence(X_ae, AE_SEQ_LEN)
    if seq is None:
        return {
            "score": None,
            "status": "Not enough data",
            "warning_th": warning_th,
            "critical_th": critical_th
        }

    X_pred = ae_model.predict(seq, verbose=0)
    error = float(np.mean((seq - X_pred) ** 2))

    if error > critical_th:
        status = "Critical"
    elif error > warning_th:
        status = "Warning"
    else:
        status = "Normal"

    return {
        "score": error,
        "status": status,
        "warning_th": warning_th,
        "critical_th": critical_th
    }

# =====================
# GMAIL ALERT
# =====================

def send_gmail_alert(subject, body):
    gmail_user = os.getenv("GMAIL_USER")
    gmail_app_password = os.getenv("GMAIL_APP_PASSWORD")
    alert_to = os.getenv("ALERT_TO", gmail_user)

    if not gmail_user or not gmail_app_password or not alert_to:
        print("Gmail alert skipped: missing GMAIL_USER / GMAIL_APP_PASSWORD / ALERT_TO")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = alert_to
    msg.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(gmail_user, gmail_app_password)
        server.send_message(msg)

    return True

# =====================
# RUNTIME STATE
# =====================

ENGINE_RUNTIME = {}
RUNTIME_LOCK = threading.Lock()

for eid in sorted(df["engine_id"].unique(), key=lambda x: int(x)):
    ENGINE_RUNTIME[str(eid)] = {
        "engine_id": str(eid),
        "cycle": 1,
        "running": False,
        "speed_ms": 30000,
        "last_tick": time.time(),
        "ae_status": "Not enough data",
        "anomaly_score": None,
        "latest_rul": None,
        "max_cycle": int(len(df[df["engine_id"] == str(eid)])),
        "last_alert_key": None
    }

def save_runtime_state():
    try:
        serializable = {}
        for eid, state in ENGINE_RUNTIME.items():
            serializable[eid] = {
                "engine_id": str(state["engine_id"]),
                "cycle": int(state["cycle"]),
                "running": bool(state["running"]),
                "speed_ms": int(state.get("speed_ms", 30000)),
                "ae_status": state.get("ae_status"),
                "anomaly_score": state.get("anomaly_score"),
                "latest_rul": state.get("latest_rul"),
                "max_cycle": int(state["max_cycle"]),
                "last_alert_key": state.get("last_alert_key")
            }

        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("ERROR save_runtime_state:", e)

def load_runtime_state():
    if not os.path.exists(STATE_PATH):
        return

    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)

        for eid, saved_state in saved.items():
            if eid in ENGINE_RUNTIME:
                ENGINE_RUNTIME[eid]["cycle"] = int(saved_state.get("cycle", 1))
                ENGINE_RUNTIME[eid]["running"] = bool(saved_state.get("running", False))
                ENGINE_RUNTIME[eid]["speed_ms"] = int(saved_state.get("speed_ms", 30000))
                ENGINE_RUNTIME[eid]["ae_status"] = saved_state.get("ae_status", "Not enough data")
                ENGINE_RUNTIME[eid]["anomaly_score"] = saved_state.get("anomaly_score", None)
                ENGINE_RUNTIME[eid]["latest_rul"] = saved_state.get("latest_rul", None)
                ENGINE_RUNTIME[eid]["last_alert_key"] = saved_state.get("last_alert_key", None)
                ENGINE_RUNTIME[eid]["last_tick"] = time.time()
    except Exception as e:
        print("ERROR load_runtime_state:", e)

def maybe_send_engine_alert(engine_id):
    state = ENGINE_RUNTIME[engine_id]
    status = state.get("ae_status")
    score = state.get("anomaly_score")
    cycle = state.get("cycle")

    if status not in ["Warning", "Critical"]:
        return

    alert_key = f"{engine_id}:{cycle}:{status}"
    if state.get("last_alert_key") == alert_key:
        return

    subject = f"[Engine Alert] Engine {engine_id} - {status}"
    body = (
        f"Engine: {engine_id}\n"
        f"Cycle: {cycle}\n"
        f"Status: {status}\n"
        f"Anomaly score: {score}\n"
        f"Warning threshold: {ae_config['warning_th']}\n"
        f"Critical threshold: {ae_config['critical_th']}\n"
    )

    try:
        sent = send_gmail_alert(subject, body)
        if sent:
            state["last_alert_key"] = alert_key
    except Exception as e:
        print("ERROR send_gmail_alert:", e)

def update_engine_runtime_metrics(engine_id, cycle):
    d = get_engine_data(engine_id).iloc[:cycle]
    ae_result = predict_anomaly_from_engine_df(d)

    ENGINE_RUNTIME[engine_id]["cycle"] = int(cycle)
    ENGINE_RUNTIME[engine_id]["ae_status"] = ae_result["status"]
    ENGINE_RUNTIME[engine_id]["anomaly_score"] = ae_result["score"]

    maybe_send_engine_alert(engine_id)

load_runtime_state()

def runtime_worker():
    while True:
        time.sleep(0.5)

        with RUNTIME_LOCK:
            now = time.time()

            for eid, state in ENGINE_RUNTIME.items():
                if not state["running"]:
                    continue

                max_cycle = int(state["max_cycle"])
                current_cycle = int(state["cycle"])
                speed_ms = max(1000, int(state.get("speed_ms", 30000)))
                elapsed_ms = (now - state["last_tick"]) * 1000.0

                if elapsed_ms < speed_ms:
                    continue

                step_count = int(elapsed_ms // speed_ms)
                if step_count <= 0:
                    continue

                new_cycle = min(current_cycle + step_count, max_cycle)
                consumed_sec = (step_count * speed_ms) / 1000.0
                state["last_tick"] = state["last_tick"] + consumed_sec

                if new_cycle != current_cycle:
                    update_engine_runtime_metrics(eid, new_cycle)
                    save_runtime_state()

                if new_cycle >= max_cycle:
                    state["running"] = False
                    save_runtime_state()

runtime_thread = threading.Thread(target=runtime_worker, daemon=True)
runtime_thread.start()

atexit.register(save_runtime_state)

# =====================
# PAGE ROUTES
# =====================

@app.route("/")
def home_page():
    return render_template("home.html")

@app.route("/detail")
def detail_page():
    engine_id = request.args.get("engine_id", "")
    return render_template("detail.html", engine_id=engine_id)

# =====================
# API
# =====================

@app.route("/engines", methods=["GET"])
def get_engines():
    try:
        with RUNTIME_LOCK:
            result = [
                dict(ENGINE_RUNTIME[eid])
                for eid in sorted(ENGINE_RUNTIME.keys(), key=lambda x: int(x))
            ]
        return jsonify(result)
    except Exception as e:
        print("ERROR /engines:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/engine/<engine_id>", methods=["GET"])
def engine_detail(engine_id):
    try:
        if str(engine_id) not in ENGINE_RUNTIME:
            return jsonify({"error": "Engine not found"}), 404

        cycle_param = request.args.get("cycle", default=None, type=int)

        with RUNTIME_LOCK:
            runtime_cycle = int(ENGINE_RUNTIME[str(engine_id)]["cycle"])

        cycle_to_use = cycle_param if cycle_param is not None else runtime_cycle

        d = get_engine_data(engine_id)
        if len(d) == 0:
            return jsonify({"error": "Engine not found"}), 404

        d = d.iloc[:cycle_to_use]
        if len(d) == 0:
            return jsonify({"error": "No data at this cycle"}), 404

        row = d.tail(1).iloc[0]
        ae_result = predict_anomaly_from_engine_df(d)

        ops = {col: float(row[col]) for col in op_cols if col in row.index}
        sensors = {col: float(row[col]) for col in sensor_cols if col in row.index}

        with RUNTIME_LOCK:
            ENGINE_RUNTIME[str(engine_id)]["cycle"] = int(len(d))
            ENGINE_RUNTIME[str(engine_id)]["ae_status"] = ae_result["status"]
            ENGINE_RUNTIME[str(engine_id)]["anomaly_score"] = ae_result["score"]
            save_runtime_state()

        return jsonify({
            "engine_id": str(engine_id),
            "cycle": int(len(d)),
            "ops": ops,
            "sensors": sensors,
            "ae_status": ae_result["status"],
            "anomaly_score": ae_result["score"],
            "warning_th": ae_result["warning_th"],
            "critical_th": ae_result["critical_th"],
            "ae_seq_len": int(AE_SEQ_LEN),
            "ae_ready": len(d) >= int(AE_SEQ_LEN),
            "ae_remaining": max(0, int(AE_SEQ_LEN) - len(d))
        })

    except Exception as e:
        print(f"ERROR /engine/{engine_id}:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/engine_state", methods=["POST"])
def update_engine_state():
    try:
        data = request.json
        engine_id = str(data.get("engine_id", ""))
        running = data.get("running", None)
        cycle = data.get("cycle", None)
        reset = data.get("reset", False)
        speed_ms = data.get("speed_ms", None)

        if engine_id not in ENGINE_RUNTIME:
            return jsonify({"error": "Engine not found"}), 404

        with RUNTIME_LOCK:
            if reset:
                ENGINE_RUNTIME[engine_id]["cycle"] = 1
                ENGINE_RUNTIME[engine_id]["running"] = False
                ENGINE_RUNTIME[engine_id]["latest_rul"] = None
                ENGINE_RUNTIME[engine_id]["anomaly_score"] = None
                ENGINE_RUNTIME[engine_id]["ae_status"] = "Not enough data"
                ENGINE_RUNTIME[engine_id]["last_tick"] = time.time()
                ENGINE_RUNTIME[engine_id]["speed_ms"] = 30000
                ENGINE_RUNTIME[engine_id]["last_alert_key"] = None
            else:
                if cycle is not None:
                    ENGINE_RUNTIME[engine_id]["cycle"] = int(cycle)

                if speed_ms is not None:
                    ENGINE_RUNTIME[engine_id]["speed_ms"] = int(speed_ms)

                if running is not None:
                    ENGINE_RUNTIME[engine_id]["running"] = bool(running)
                    ENGINE_RUNTIME[engine_id]["last_tick"] = time.time()

            current_cycle = int(ENGINE_RUNTIME[engine_id]["cycle"])
            update_engine_runtime_metrics(engine_id, current_cycle)
            save_runtime_state()

            state = dict(ENGINE_RUNTIME[engine_id])

        return jsonify({"ok": True, "state": state})
    except Exception as e:
        print("ERROR /engine_state:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/predict_rul", methods=["POST"])
def predict_rul():
    try:
        data = request.json
        engine_id = str(data["engine_id"])
        cycle = int(data["cycle"])

        d = get_engine_data(engine_id).iloc[:cycle]
        result = predict_rul_from_engine_df(d)

        if result["rul"] is None:
            return jsonify({"error": result["message"]}), 400

        with RUNTIME_LOCK:
            if engine_id in ENGINE_RUNTIME:
                ENGINE_RUNTIME[engine_id]["latest_rul"] = float(result["rul"])
                ENGINE_RUNTIME[engine_id]["cycle"] = cycle
                save_runtime_state()

        return jsonify(result)

    except Exception as e:
        print("ERROR /predict_rul:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/anomaly", methods=["POST"])
def anomaly():
    try:
        data = request.json
        engine_id = str(data["engine_id"])
        cycle = int(data["cycle"])

        d = get_engine_data(engine_id).iloc[:cycle]
        result = predict_anomaly_from_engine_df(d)

        return jsonify(result)

    except Exception as e:
        print("ERROR /anomaly:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)