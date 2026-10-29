import os
from openai import OpenAI

PROVIDERS = {
    "ollama": (os.getenv("OLLAMA_BASE_URL", "http://10.0.1.21:11434/v1"), "ollama", "llama3.2:3b"),
    "groq": ("https://api.groq.com/openai/v1", os.getenv("GROQ_API_KEY", ""), "openai/gpt-oss-120b"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", os.getenv("GEMINI_API_KEY", ""), "gemini-flash-latest"),
    "github": ("https://models.github.ai/inference", os.getenv("GITHUB_MODELS_TOKEN", ""), "openai/gpt-4o-mini"),
}

for name, (base, key, model) in PROVIDERS.items():
    if name != "ollama" and not key.strip():
        print(f"{name:8} SKIP (no key)")
        continue
    try:
        c = OpenAI(base_url=base, api_key=(key.strip() or "ollama"), timeout=25)
        r = c.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            max_tokens=400,
        )
        print(f"{name:8} OK -> {r.choices[0].message.content.strip()!r} ({model})")
    except Exception as e:
        print(f"{name:8} ERROR -> {type(e).__name__}: {str(e)[:160]}")
