"""LLM + embeddings layer built on LangChain (langchain-ollama).
Every function returns None on failure so callers fall back to rules (the demo never breaks).
If langchain-ollama is not installed, chat() falls back to the plain `ollama` client."""
try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
except ImportError:
    ChatOllama = OllamaEmbeddings = None
try:
    import ollama
except ImportError:
    ollama = None

CONFIG = {"use_llm": True, "model": "llama3.1:8b", "embed_model": "nomic-embed-text"}


def backend():
    if ChatOllama is not None:
        return "LangChain ChatOllama"
    return "ollama client" if ollama is not None else "none (rules only)"


def chat(system, user, json_mode=False):
    """Plain or JSON-mode chat completion."""
    if not CONFIG["use_llm"]:
        return None
    try:
        if ChatOllama is not None:
            m = ChatOllama(model=CONFIG["model"], temperature=0, format="json" if json_mode else None)
            return m.invoke([("system", system), ("human", user)]).content
        if ollama is not None:
            kw = {"format": "json"} if json_mode else {}
            r = ollama.chat(model=CONFIG["model"], options={"temperature": 0}, **kw,
                            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            return r["message"]["content"]
    except Exception:
        return None
    return None


def tool_call(system, user, tools):
    """Native tool calling: the model is handed real tool schemas and returns (text, [tool_calls])."""
    if not CONFIG["use_llm"] or ChatOllama is None:
        return None
    try:
        r = ChatOllama(model=CONFIG["model"], temperature=0).bind_tools(tools).invoke(
            [("system", system), ("human", user)])
        return (r.content or ""), list(r.tool_calls or [])
    except Exception:
        return None


def embedder():
    """Embedding model for the Chroma vector store (needs: ollama pull nomic-embed-text)."""
    if not CONFIG["use_llm"] or OllamaEmbeddings is None:
        return None
    return OllamaEmbeddings(model=CONFIG["embed_model"])
