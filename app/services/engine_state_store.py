import time
from collections import defaultdict
from typing import Any

import numpy as np
_ENGINE_STORE = None


def set_engine_store(store):
    global _ENGINE_STORE
    _ENGINE_STORE = store


def get_engine_store():
    return _ENGINE_STORE


def get_engine_state(engine_id):
    if _ENGINE_STORE is None:
        return None
    return _ENGINE_STORE.get_engine_state(engine_id)


def get_all_engine_states():
    if _ENGINE_STORE is None:
        return {}
    return _ENGINE_STORE.get_all_engine_states()

class EngineStateStore:
    def __init__(
        self,
        df,
        rf_model,
        rf_scaler,
        rf_feature_cols,
        gru_model,
        gru_scaler,
        gru_feature_cols,
        gru_seq_len,
        ae_model,
        ae_scaler,
        ae_feature_cols,
        ae_seq_len,
        warning_th,
        critical_th,
        op_cols,
        sensor_cols,
    ):
        self.df = df.copy()

        self.rf_model = rf_model
        self.rf_scaler = rf_scaler
        self.rf_feature_cols = list(rf_feature_cols)

        self.gru_model = gru_model
        self.gru_scaler = gru_scaler
        self.gru_feature_cols = list(gru_feature_cols)
        self.gru_seq_len = int(gru_seq_len)

        self.ae_model = ae_model
        self.ae_scaler = ae_scaler
        self.ae_feature_cols = list(ae_feature_cols)
        self.ae_seq_len = int(ae_seq_len)

        self.warning_th = float(warning_th)
        self.critical_th = float(critical_th)

        self.op_cols = list(op_cols)
        self.sensor_cols = list(sensor_cols)

        self.sensor_baseline = self._build_sensor_baseline()
        self.alert_logs = defaultdict(list)
        self.engine_states = {}

        self._init_engine_states()

    # =====================
    # INIT
    # =====================

    def _build_sensor_baseline(self) -> dict[str, float]:
        baseline = {}
        for col in self.sensor_cols:
            if col in self.df.columns:
                baseline[col] = float(self.df[col].mean())
        return baseline

    def _init_engine_states(self) -> None:
        engine_ids = sorted(self.df["engine_id"].astype(str).unique(), key=lambda x: int(x))

        for eid in engine_ids:
            max_cycle = int(len(self.get_engine_data(eid)))
            self.engine_states[str(eid)] = {
                "engine_id": str(eid),
                "cycle": 1,
                "running": False,
                "speed_ms": 30000,
                "last_tick": time.time(),
                "latest_rul": None,
                "rul_model": None,
                "anomaly_score": None,
                "ae_status": "Not enough data",
                "health_score": None,
                "abnormal_sensors": [],
                "recent_alerts": [],
                "rul_last_5": [],
                "anomaly_scores_last_5": [],
                "max_cycle": max_cycle,
                "last_alert_key": None,
                "last_updated_at": None,
            }

    # =====================
    # BASIC GET
    # =====================

    def get_engine_data(self, engine_id: str):
        return self.df[self.df["engine_id"].astype(str) == str(engine_id)].copy()

    def get_engine_state(self, engine_id: str) -> dict[str, Any] | None:
        return self.engine_states.get(str(engine_id))

    def get_all_engine_states(self) -> dict[str, dict[str, Any]]:
        return self.engine_states

    # =====================
    # FEATURE UTILS
    # =====================

    def prepare_features_for_model(self, df_engine, feature_cols, scaler, model_name="model"):
        missing_cols = [col for col in feature_cols if col not in df_engine.columns]
        if missing_cols:
            raise ValueError(f"{model_name} missing feature columns: {missing_cols}")

        X = df_engine[feature_cols].copy()
        return scaler.transform(X)

    @staticmethod
    def create_sequence(X, seq_len):
        if len(X) < seq_len:
            return None
        return np.array([X[-seq_len:]], dtype=np.float32)

    # =====================
    # RUL
    # =====================

    def predict_rul_from_engine_df(self, df_engine) -> dict[str, Any]:
        if len(df_engine) == 0:
            return {
                "rul": None,
                "model": None,
                "message": "No data"
            }

        X_rf = self.prepare_features_for_model(
            df_engine,
            self.rf_feature_cols,
            self.rf_scaler,
            model_name="RF"
        )

        # chưa đủ chuỗi cho GRU => RF
        if len(X_rf) < self.gru_seq_len:
            pred = self.rf_model.predict(X_rf[-1].reshape(1, -1))[0]
            return {
                "rul": float(pred),
                "model": "RF",
                "message": f"Chưa đủ {self.gru_seq_len} bước, dùng Random Forest"
            }

        X_gru = self.prepare_features_for_model(
            df_engine,
            self.gru_feature_cols,
            self.gru_scaler,
            model_name="GRU"
        )

        seq = self.create_sequence(X_gru, self.gru_seq_len)
        if seq is None:
            pred = self.rf_model.predict(X_rf[-1].reshape(1, -1))[0]
            return {
                "rul": float(pred),
                "model": "RF",
                "message": "Không tạo được sequence GRU, fallback Random Forest"
            }

        pred = self.gru_model.predict(seq, verbose=0)[0][0]
        return {
            "rul": float(pred),
            "model": "GRU",
            "message": f"Đủ chuỗi {self.gru_seq_len} bước, dùng GRU"
        }

    # =====================
    # ANOMALY
    # =====================

    def predict_anomaly_from_engine_df(self, df_engine) -> dict[str, Any]:
        if len(df_engine) == 0:
            return {
                "score": None,
                "status": "No data",
                "warning_th": self.warning_th,
                "critical_th": self.critical_th
            }

        X_ae = self.prepare_features_for_model(
            df_engine,
            self.ae_feature_cols,
            self.ae_scaler,
            model_name="AE"
        )

        seq = self.create_sequence(X_ae, self.ae_seq_len)
        if seq is None:
            return {
                "score": None,
                "status": "Not enough data",
                "warning_th": self.warning_th,
                "critical_th": self.critical_th
            }

        X_pred = self.ae_model.predict(seq, verbose=0)
        error = float(np.mean((seq - X_pred) ** 2))

        if error > self.critical_th:
            status = "Critical"
        elif error > self.warning_th:
            status = "Warning"
        else:
            status = "Normal"

        return {
            "score": error,
            "status": status,
            "warning_th": self.warning_th,
            "critical_th": self.critical_th
        }

    # =====================
    # HEALTH / SENSOR ANALYSIS
    # =====================

    def compute_health_score(self, rul_value, anomaly_score):
        if rul_value is None and anomaly_score is None:
            return None

        rul_part = 0.0 if rul_value is None else min(max(float(rul_value), 0.0), 100.0)
        anomaly_penalty = 0.0 if anomaly_score is None else min(max(float(anomaly_score), 0.0), 2.0) * 50.0

        score = int(max(0, min(100, rul_part - anomaly_penalty)))
        return score

    def compute_abnormal_sensors(self, row, top_k=3) -> list[str]:
        deviations = []

        for col in self.sensor_cols:
            if col not in row.index:
                continue

            baseline = self.sensor_baseline.get(col)
            if baseline is None:
                continue

            try:
                value = float(row[col])
                diff = abs(value - baseline)
                deviations.append((col, diff))
            except Exception:
                continue

        deviations.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in deviations[:top_k]]

    # =====================
    # ALERTS
    # =====================

    def append_alert(self, engine_id: str, cycle: int, status: str) -> None:
        if status not in ["Warning", "Critical"]:
            return

        alert_text = f"cycle {cycle}: {status.lower()}"
        logs = self.alert_logs[str(engine_id)]

        if not logs or logs[-1] != alert_text:
            logs.append(alert_text)

        if len(logs) > 20:
            self.alert_logs[str(engine_id)] = logs[-20:]

    # =====================
    # STATE UPDATE
    # =====================

    def update_engine_runtime_metrics(self, engine_id: str, cycle: int) -> dict[str, Any] | None:
        engine_id = str(engine_id)
        df_engine = self.get_engine_data(engine_id).iloc[:cycle]

        if len(df_engine) == 0:
            return None

        row = df_engine.tail(1).iloc[0]

        rul_result = self.predict_rul_from_engine_df(df_engine)
        ae_result = self.predict_anomaly_from_engine_df(df_engine)

        rul_value = rul_result["rul"]
        anomaly_score = ae_result["score"]
        ae_status = ae_result["status"]

        abnormal_sensors = self.compute_abnormal_sensors(row, top_k=3)
        health_score = self.compute_health_score(rul_value, anomaly_score)

        state = self.engine_states[engine_id]

        if rul_value is not None:
            state["rul_last_5"].append(round(float(rul_value), 2))
            state["rul_last_5"] = state["rul_last_5"][-5:]

        if anomaly_score is not None:
            state["anomaly_scores_last_5"].append(round(float(anomaly_score), 4))
            state["anomaly_scores_last_5"] = state["anomaly_scores_last_5"][-5:]

        self.append_alert(engine_id, cycle, ae_status)

        state["cycle"] = int(cycle)
        state["latest_rul"] = None if rul_value is None else round(float(rul_value), 2)
        state["rul_model"] = rul_result["model"]
        state["anomaly_score"] = None if anomaly_score is None else round(float(anomaly_score), 6)
        state["ae_status"] = ae_status
        state["health_score"] = health_score
        state["abnormal_sensors"] = abnormal_sensors
        state["recent_alerts"] = self.alert_logs[engine_id][-5:]
        state["last_updated_at"] = time.time()

        return state

    # =====================
    # DETAIL PAYLOAD
    # =====================

    def build_engine_detail_payload(self, engine_id: str, cycle: int | None = None) -> dict[str, Any] | None:
        engine_id = str(engine_id)

        if engine_id not in self.engine_states:
            return None

        runtime_cycle = int(self.engine_states[engine_id]["cycle"])
        cycle_to_use = int(cycle) if cycle is not None else runtime_cycle

        df_engine = self.get_engine_data(engine_id).iloc[:cycle_to_use]
        if len(df_engine) == 0:
            return None

        row = df_engine.tail(1).iloc[0]

        ops = {}
        sensors = {}

        for col in self.op_cols:
            if col in row.index:
                ops[col] = float(row[col])

        for col in self.sensor_cols:
            if col in row.index:
                sensors[col] = float(row[col])

        self.update_engine_runtime_metrics(engine_id, len(df_engine))
        state = self.engine_states[engine_id]

        return {
            "engine_id": engine_id,
            "cycle": int(len(df_engine)),
            "ops": ops,
            "sensors": sensors,
            "ae_status": state["ae_status"],
            "anomaly_score": state["anomaly_score"],
            "warning_th": self.warning_th,
            "critical_th": self.critical_th,
            "ae_seq_len": int(self.ae_seq_len),
            "ae_ready": len(df_engine) >= int(self.ae_seq_len),
            "ae_remaining": max(0, int(self.ae_seq_len) - len(df_engine)),
            "latest_rul": state["latest_rul"],
            "rul_model": state["rul_model"],
            "health_score": state["health_score"],
            "abnormal_sensors": state["abnormal_sensors"],
            "recent_alerts": state["recent_alerts"],
            "rul_last_5": state["rul_last_5"],
            "anomaly_scores_last_5": state["anomaly_scores_last_5"],
        }