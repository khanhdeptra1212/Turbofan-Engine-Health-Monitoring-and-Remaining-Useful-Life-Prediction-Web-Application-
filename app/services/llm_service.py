import json
import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"


def ask_local_llm(prompt: str, system: str | None = None, timeout: int = 40) -> str:
    final_system = system or (
        "Reply in the same language as the user. "
        "Be natural, concise, context-aware, and do not invent data."
    )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "system": final_system,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 300
            }
        },
        timeout=timeout
    )
    response.raise_for_status()
    data = response.json()
    return (data.get("response") or "").strip()


def ask_local_llm_json(prompt: str, system: str | None = None, timeout: int = 40) -> dict:
    final_system = system or (
        "Return valid JSON only. "
        "Do not wrap in markdown. "
        "Do not add explanation outside JSON."
    )

    text = ask_local_llm(prompt=prompt, system=final_system, timeout=timeout)

    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError(f"Invalid JSON from model: {text}")