_session_context = {}


def update_session_context(session_id: str, **kwargs):
    if session_id not in _session_context:
        _session_context[session_id] = {}

    for key, value in kwargs.items():
        if value is not None:
            _session_context[session_id][key] = value


def get_session_context(session_id: str):
    return _session_context.get(session_id, {})