import json

from app.services.context_builder import build_context
from app.services.memory_service import get_recent_memory, save_memory
from app.services.session_context_service import (
    get_session_context,
    update_session_context,
)
from app.services.llm_service import ask_local_llm, ask_local_llm_json


BASE_SYSTEM = """
Bạn là trợ lý AI trong hệ thống giám sát động cơ.

Quy tắc:
- Trả lời bằng đúng ngôn ngữ người dùng đang dùng.
- Với kiến thức chung ngoài bài toán engine của người dùng, hãy trả lời như chatbot bình thường, theo ngữ cảnh hội thoại.
- Chỉ khi câu hỏi thực sự liên quan đến hệ thống engine / dữ liệu dự án / trạng thái engine / RUL / anomaly / warning / critical / so sánh engine thì mới được dùng dữ liệu của hệ thống.
- Nếu dùng dữ liệu hệ thống thì không được bịa.
- Nếu dữ liệu không đủ thì nói rõ là dữ liệu không đủ.
""".strip()


def _save_turn(session_id: str, user_message: str, answer: str) -> str:
    save_memory(session_id, "user", user_message)
    save_memory(session_id, "assistant", answer)
    return answer


def _route_request(user_message: str, context: dict) -> dict:
    prompt = f"""
Phân loại yêu cầu người dùng thành một trong các route sau:

- general_chat
  Dùng cho mọi câu hỏi kiến thức chung, đời sống, học tập, trò chuyện, dịch thuật, lập trình, ngày giờ, giao tiếp...
  Không dùng dữ liệu hệ thống engine.

- system_overview
  Dùng khi người dùng muốn biết tổng quan hệ thống của họ, tổng số engine, số engine warning/critical/running...

- engine_detail
  Dùng khi người dùng hỏi về một engine cụ thể hoặc "engine này".

- engine_compare
  Dùng khi người dùng muốn so sánh nhiều engine.

- engine_group
  Dùng khi người dùng hỏi nhóm engine theo trạng thái như warning, critical, normal.

Trả về JSON hợp lệ với đúng format:
{{
  "route": "general_chat | system_overview | engine_detail | engine_compare | engine_group",
  "engine_ids": [danh sách id engine nếu có, có thể rỗng],
  "group_status": "warning | critical | normal | null",
  "confidence": 0.0
}}

Ngữ cảnh hội thoại gần đây:
{context["recent_memory_text"]}

Trang hiện tại:
{context["current_page"]}

Engine hiện tại:
{json.dumps(context["current_engine"], ensure_ascii=False)}

Danh sách engine hiện có:
{json.dumps([e["engine_id"] for e in context["engine_states"]], ensure_ascii=False)}

Câu hỏi hiện tại:
{user_message}
""".strip()

    route = ask_local_llm_json(prompt)
    return {
        "route": route.get("route", "general_chat"),
        "engine_ids": route.get("engine_ids", []),
        "group_status": route.get("group_status"),
        "confidence": route.get("confidence", 0.0),
    }


def _select_engines_by_ids(context: dict, engine_ids: list) -> list:
    state_map = {str(e["engine_id"]): e for e in context["engine_states"]}
    return [state_map[str(eid)] for eid in engine_ids if str(eid) in state_map]


def _select_group(context: dict, group_status: str | None) -> list:
    if not group_status:
        return []
    return [
        e for e in context["engine_states"]
        if str(e.get("ae_status", "")).strip().lower() == str(group_status).strip().lower()
    ]


def handle_general_chat(user_message: str, context: dict, route_payload: dict) -> str:
    prompt = f"""
{BASE_SYSTEM}

Đây là hội thoại gần đây:
{context["recent_memory_text"]}

Câu hỏi hiện tại:
{user_message}

Hãy trả lời như chat bình thường, tự nhiên, đúng ngữ cảnh.
Không ép sang kiểu báo cáo.
Nếu là câu hỏi đơn giản thì trả lời trực tiếp.
""".strip()
    return ask_local_llm(prompt)


def handle_system_overview(user_message: str, context: dict, route_payload: dict) -> str:
    prompt = f"""
{BASE_SYSTEM}

Dữ liệu overview hệ thống engine:
{json.dumps(context["engine_states"], ensure_ascii=False, indent=2)}

Câu hỏi:
{user_message}

Hãy trả lời đúng theo dữ liệu này.
Nếu người dùng hỏi "tổng quan hệ thống của tôi" thì phải hiểu là tổng quan hệ thống engine trong dự án của họ.
Trả lời đầy đủ nhưng gọn, đúng ngôn ngữ người dùng.
""".strip()
    return ask_local_llm(prompt)


def handle_engine_detail(user_message: str, context: dict, route_payload: dict) -> str:
    selected = _select_engines_by_ids(context, route_payload.get("engine_ids", []))
    engine = selected[0] if selected else context.get("current_engine")

    prompt = f"""
{BASE_SYSTEM}

Dữ liệu engine cần trả lời:
{json.dumps(engine, ensure_ascii=False, indent=2)}

Câu hỏi:
{user_message}

Hãy trả lời đúng theo dữ liệu engine này.
Nêu rõ các trường quan trọng nếu có: cycle, running, latest_rul, ae_status, anomaly_score, health_score, abnormal_sensors.
Không bịa.
""".strip()
    return ask_local_llm(prompt)


def handle_engine_compare(user_message: str, context: dict, route_payload: dict) -> str:
    selected = _select_engines_by_ids(context, route_payload.get("engine_ids", []))

    prompt = f"""
{BASE_SYSTEM}

Dữ liệu các engine cần so sánh:
{json.dumps(selected, ensure_ascii=False, indent=2)}

Câu hỏi:
{user_message}

Hãy so sánh chính xác dựa trên dữ liệu trên.
Không bịa.
""".strip()
    return ask_local_llm(prompt)


def handle_engine_group(user_message: str, context: dict, route_payload: dict) -> str:
    selected = _select_group(context, route_payload.get("group_status"))

    prompt = f"""
{BASE_SYSTEM}

Dữ liệu nhóm engine:
{json.dumps(selected, ensure_ascii=False, indent=2)}

Trạng thái nhóm:
{route_payload.get("group_status")}

Câu hỏi:
{user_message}

Hãy trả lời chính xác dựa trên dữ liệu trên.
Nếu không có engine nào thì nói rõ là không có.
""".strip()
    return ask_local_llm(prompt)


HANDLERS = {
    "general_chat": handle_general_chat,
    "system_overview": handle_system_overview,
    "engine_detail": handle_engine_detail,
    "engine_compare": handle_engine_compare,
    "engine_group": handle_engine_group,
}


def generate_chat_answer(
    user_message: str,
    session_id: str = "default",
    current_page: str | None = None,
    current_engine_id: int | None = None
) -> str:
    payload = {
        "current_page": current_page,
        "current_engine_id": current_engine_id,
    }
    clean_payload = {k: v for k, v in payload.items() if v is not None}
    clean_payload and update_session_context(session_id, **clean_payload)

    memory = get_recent_memory(session_id)
    session_context = get_session_context(session_id)
    context = build_context(user_message, memory, session_context)

    route_payload = _route_request(user_message, context)
    handler = HANDLERS.get(route_payload.get("route"), handle_general_chat)
    answer = handler(user_message, context, route_payload)

    return _save_turn(session_id, user_message, answer)