from app.services.engine_state_store import get_all_engine_states


def _recent_memory_as_text(memory: list, limit: int = 6) -> str:
    return "\n".join(
        f"{item.get('role', 'user')}: {item.get('content', '')}"
        for item in memory[-limit:]
    )


def _compact_engine_state(engine: dict) -> dict:
    return {
        "engine_id": engine.get("engine_id"),
        "running": engine.get("running"),
        "cycle": engine.get("cycle"),
        "latest_rul": engine.get("latest_rul"),
        "rul_model": engine.get("rul_model"),
        "ae_status": engine.get("ae_status"),
        "anomaly_score": engine.get("anomaly_score"),
        "health_score": engine.get("health_score"),
        "abnormal_sensors": engine.get("abnormal_sensors", []),
        "recent_alerts": engine.get("recent_alerts", []),
        "rul_last_5": engine.get("rul_last_5", []),
        "anomaly_scores_last_5": engine.get("anomaly_scores_last_5", []),
    }


def build_context(user_message: str, memory: list, session_context: dict | None = None):
    all_states = get_all_engine_states() or {}
    compact_states = [_compact_engine_state(v) for v in all_states.values()]

    current_engine_id = None
    if session_context:
        current_engine_id = session_context.get("current_engine_id")

    current_engine = None
    if current_engine_id is not None and str(current_engine_id) in all_states:
        current_engine = _compact_engine_state(all_states[str(current_engine_id)])

    return {
        "user_message": user_message,
        "recent_memory_text": _recent_memory_as_text(memory),
        "current_page": (session_context or {}).get("current_page"),
        "current_engine_id": current_engine_id,
        "current_engine": current_engine,
        "engine_states": compact_states,
    }