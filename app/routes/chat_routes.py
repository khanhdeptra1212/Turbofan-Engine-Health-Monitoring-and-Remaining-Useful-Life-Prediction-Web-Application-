from flask import Blueprint, request, jsonify
from app.services.chat_service import generate_chat_answer

chat_bp = Blueprint("chat_bp", __name__)


@chat_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")
    current_page = data.get("current_page")
    current_engine_id = data.get("current_engine_id")

    if not message:
        return jsonify({"answer": "Bạn chưa nhập câu hỏi."}), 400

    try:
        if current_engine_id not in (None, "", "null"):
            current_engine_id = int(current_engine_id)
        else:
            current_engine_id = None
    except Exception:
        current_engine_id = None

    answer = generate_chat_answer(
        user_message=message,
        session_id=session_id,
        current_page=current_page,
        current_engine_id=current_engine_id
    )

    return jsonify({"answer": answer})
