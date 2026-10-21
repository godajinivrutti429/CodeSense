"""CodeSense — Streamlit chat UI for the codebase assistant."""
import json
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="CodeSense", page_icon="🧠", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = "default"
if "token" not in st.session_state:
    st.session_state.token = None


def auth_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}


def fetch_collections():
    try:
        resp = requests.get(f"{API_URL}/collections", headers=auth_headers(), timeout=10)
        if resp.status_code == 200:
            return resp.json().get("collections", [])
    except requests.RequestException:
        pass
    return []


def render_sources(sources):
    with st.expander(f"📎 Sources ({len(sources)})"):
        for s in sources:
            loc = f"{s['file_path']} (lines {s['line_start']}-{s['line_end']})"
            if s.get("function_name"):
                loc += f" — `{s['function_name']}`"
            st.markdown(f"**{loc}** — score `{s['score']}`")
            st.code(s["snippet"], language="python")


st.title("🧠 CodeSense — Chat with your Codebase")

# ---------------- Sidebar: repo management ----------------
with st.sidebar:
    st.header("📚 Indexed Repositories")
    existing = fetch_collections()

    if existing:
        if st.session_state.collection not in existing:
            st.session_state.collection = existing[0]
        chosen = st.selectbox("Active repo", options=existing,
                               index=existing.index(st.session_state.collection))
        st.session_state.collection = chosen

        if st.button("🗑️ Delete this repo", use_container_width=True):
            try:
                resp = requests.delete(f"{API_URL}/collections/{chosen}",
                                        headers=auth_headers(), timeout=30)
                if resp.status_code == 200:
                    st.success(f"Deleted '{chosen}'.")
                    st.session_state.collection = "default"
                    st.rerun()
                else:
                    st.error(resp.json().get("detail", resp.text))
            except requests.RequestException as e:
                st.error(f"Could not reach backend: {e}")
    else:
        st.caption("No repos indexed yet — add one below.")

    st.divider()
    st.header("📂 Index a New Repository")
    repo_url = st.text_input("Git URL (optional)", placeholder="https://github.com/user/repo")
    repo_path = st.text_input("...or local path on server", placeholder="/repos/myrepo")
    collection_name = st.text_input("Project name", value="default")

    if st.button("Ingest Repository", type="primary", use_container_width=True):
        if not repo_url and not repo_path:
            st.error("Provide a Git URL or a local path.")
        else:
            with st.spinner("Parsing, embedding, and indexing... this can take a while for large repos."):
                try:
                    resp = requests.post(
                        f"{API_URL}/ingest",
                        json={"repo_url": repo_url or None, "repo_path": repo_path or None,
                              "collection_name": collection_name},
                        headers=auth_headers(),
                        timeout=600,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.collection = collection_name
                        st.success(f"Indexed {data['files_processed']} files "
                                   f"→ {data['chunks_indexed']} chunks "
                                   f"(embedding model: {data['embedding_model']}).")
                        st.rerun()
                    else:
                        st.error(f"Ingestion failed: {resp.json().get('detail', resp.text)}")
                except requests.RequestException as e:
                    st.error(f"Could not reach backend: {e}")

    st.divider()
    st.caption(f"API: {API_URL}")
    st.caption(f"Active repo: **{st.session_state.collection}**")
    try:
        h = requests.get(f"{API_URL}/health", timeout=5).json()
        if not h.get("ollama_reachable"):
            st.caption("Local LLM: 🔴 Ollama unreachable — run `ollama serve`")
        elif not h.get("model_available", True):
            model_name = h.get("model", "the configured model")
            st.caption(f"Local LLM: 🟡 Ollama is up, but '{model_name}' isn't pulled — "
                       f"run `ollama pull {model_name}`")
        else:
            st.caption(f"Local LLM: 🟢 online ({h.get('model', '')})")
    except requests.RequestException:
        st.caption("Local LLM: 🔴 backend unreachable")

# ---------------- Main: chat ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

question = st.chat_input("Ask something about the codebase, e.g. 'Where is auth handled?'")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]

    with st.chat_message("assistant"):
        sources_holder = {"sources": []}

        def token_generator(resp):
            """Consume the ndjson stream: first line is sources, rest are answer tokens."""
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk["type"] == "sources":
                    sources_holder["sources"] = chunk["sources"]
                elif chunk["type"] == "token":
                    yield chunk["text"]

        try:
            resp = requests.post(
                f"{API_URL}/query/stream",
                json={
                    "collection_name": st.session_state.collection,
                    "question": question,
                    "history": history,
                    "top_k": 5,
                },
                headers=auth_headers(),
                stream=True,
                # (connect_timeout, read_timeout) — read timeout must exceed the
                # backend's own OLLAMA_READ_TIMEOUT (600s default), since a cold
                # Ollama model load can delay the very first streamed token.
                timeout=(10, 620),
            )
            if resp.status_code == 200:
                full_answer = st.write_stream(token_generator(resp))
                sources = sources_holder["sources"]
                if sources:
                    render_sources(sources)
                st.session_state.messages.append({
                    "role": "assistant", "content": full_answer, "sources": sources,
                })
            else:
                err = resp.json().get("detail", resp.text)
                st.error(f"Query failed: {err}")
        except requests.RequestException as e:
            st.error(f"Could not reach backend: {e}")