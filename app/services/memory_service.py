from collections import defaultdict, deque

_chat_memory = defaultdict(lambda: deque(maxlen=6))


def get_recent_memory(session_id: str):
    return list(_chat_memory[session_id])


def save_memory(session_id: str, role: str, content: str):
    _chat_memory[session_id].append({
        "role": role,
        "content": content,
    })