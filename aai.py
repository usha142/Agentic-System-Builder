import os
import json
import hashlib
import sqlite3
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple

import streamlit as st

# --- Hugging Face & LLM ---
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# --- Tools ---
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

# =============================================================================
# DATABASE
# =============================================================================
DB_PATH = "automation.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS cache (
            input_hash TEXT PRIMARY KEY,
            model_name TEXT,
            output TEXT,
            timestamp TIMESTAMP
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS tool_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT,
            tool_name TEXT,
            params TEXT,
            result TEXT,
            timestamp TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()


def db_cache_get(input_hash, model_name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT output FROM cache WHERE input_hash=? AND model_name=?", (input_hash, model_name))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def db_cache_set(input_hash, model_name, output):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?)",
              (input_hash, model_name, output, datetime.now()))
    conn.commit()
    conn.close()


def db_tool_log(agent_name, tool_name, params, result):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO tool_logs (agent_name, tool_name, params, result, timestamp) VALUES (?, ?, ?, ?, ?)",
              (agent_name, tool_name, json.dumps(params), result, datetime.now()))
    conn.commit()
    conn.close()


# =============================================================================
# MODEL MANAGER
# =============================================================================
@st.cache_resource
def load_local_models():
    gen_model = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(gen_model, cache_dir="./models")
    model = AutoModelForSeq2SeqLM.from_pretrained(gen_model, cache_dir="./models")
    pipeline_gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

    router_model = "distilbert-base-uncased-finetuned-sst-2-english"
    r_tok = AutoTokenizer.from_pretrained(router_model, cache_dir="./models")
    r_model = AutoModelForSequenceClassification.from_pretrained(router_model, cache_dir="./models")
    router_pipeline = pipeline("text-classification", model=r_model, tokenizer=r_tok, device=-1)

    embed = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")

    return pipeline_gen, router_pipeline, embed


class ModelManager:
    def __init__(self, key: str):
        genai.configure(api_key=key)
        self.gen_pipeline, _, _ = load_local_models()

    def _gemini(self, prompt: str):
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt, request_options={"timeout": 30})
        return response.text.strip()

    def _local(self, prompt: str):
        out = self.gen_pipeline(f"Instruction: {prompt}\nOutput:", max_length=512)
        return out[0]['generated_text']

    def generate(self, prompt: str, model="gemini"):
        key = hashlib.sha256((model + prompt).encode()).hexdigest()

        cached = db_cache_get(key, model)
        if cached:
            return cached

        try:
            out = self._gemini(prompt) if model == "gemini" else self._local(prompt)
        except:
            out = self._local(prompt)

        db_cache_set(key, model, out)
        return out


# =============================================================================
# TOOLS
# =============================================================================
class ToolRegistry:
    def __init__(self):
        self.tools = {
            "web_search": self.web_search,
            "file_write": self.file_write,
            "file_read": self.file_read,
        }

    def execute(self, tool, params, agent):
        try:
            result = self.tools[tool](**params)
        except Exception as e:
            result = f"TOOL ERROR: {str(e)}"
        db_tool_log(agent, tool, params, str(result))
        return result

    def web_search(self, query, max_results=3):
        results = []

        if DDGS_AVAILABLE:
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=max_results))
                    if results:
                        return json.dumps(results, indent=2)
            except:
                pass

        try:
            import wikipedia
            summary = wikipedia.summary(query, sentences=5)
            return json.dumps([{
                "title": query,
                "body": summary,
                "source": "wikipedia"
            }], indent=2)
        except:
            pass

        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(
                f"Provide factual search results for: {query}",
                request_options={"timeout": 15}
            )
            return json.dumps([{
                "title": "AI Generated Result",
                "body": response.text.strip(),
                "source": "gemini"
            }], indent=2)
        except:
            pass

        return json.dumps([{
            "title": "Search Failed",
            "body": f"No results available for query: {query}",
            "source": "fallback"
        }], indent=2)

    def file_write(self, path, content):
        workspace = os.path.abspath("workspace")
        os.makedirs(workspace, exist_ok=True)
        full = os.path.join(workspace, path)

        with open(full, "w", encoding="utf-8") as f:
            f.write(content)

        return f"written: {full} | size={len(content)}"

    def file_read(self, path):
        full = os.path.join(os.path.abspath("workspace"), path)
        with open(full, "r", encoding="utf-8") as f:
            return f.read()


# =============================================================================
# AGENT
# =============================================================================
class Agent:
    def __init__(self, name, role, instructions, tools):
        self.name = name
        self.role = role
        self.instructions = instructions
        self.tools = tools

    def run(self, context, mm, tr, state, loops=3):

        st.markdown(f"## ▶️ {self.name}")
        progress = st.progress(0)

        system = f"""
You are {self.role}.
{self.instructions}

TOOLS: {self.tools}

RULES:
- If using tool → JSON only
- Else → Answer:
"""

        convo = [system, context]

        for i in range(loops):
            progress.progress((i+1)/loops)
            st.write(f"🔁 Loop {i+1}/{loops}")

            prompt = "\n".join(convo)

            with st.expander("Prompt"):
                st.code(prompt[:1500])

            st.write("🧠 Generating...")
            t0 = time.time()

            out = mm.generate(prompt)

            st.write(f"⏱ {round(time.time()-t0,2)}s")

            with st.expander("Output", expanded=True):
                st.code(out[:1500])

            data = None

            # ---------- Handle tool_code JSON (like {"tool_code": "print(...)"}) ----------
            try:
                parsed = json.loads(out)
                if "tool_code" in parsed:
                    tool_code_str = parsed["tool_code"]
                    # Extract function call (e.g., file_read('path'))
                    call_match = re.search(r'([a-z_]+)\(([^)]+)\)', tool_code_str)
                    if call_match:
                        tool = call_match.group(1)
                        # Parse arguments (simple quoted strings)
                        args = call_match.group(2)
                        if tool == "file_read":
                            path_match = re.search(r'[\'"]([^\'"]+)[\'"]', args)
                            if path_match:
                                data = {
                                    "tool": "file_read",
                                    "params": {"path": path_match.group(1)}
                                }
                        elif tool == "web_search":
                            query_match = re.search(r'query=[\'"](.+?)[\'"]', args)
                            if query_match:
                                data = {
                                    "tool": "web_search",
                                    "params": {"query": query_match.group(1)}
                                }
            except (json.JSONDecodeError, TypeError):
                pass

            # ---------- Legacy tool_code detection (print(...)) ----------
            if not data and "tool_code" in out:
                match = re.search(r'print\((.*?)\)', out)
                if match:
                    call = match.group(1)
                    if call.startswith("file_read"):
                        path_match = re.search(r'file_read\([\'"](.+?)[\'"]\)', call)
                        if path_match:
                            data = {
                                "tool": "file_read",
                                "params": {"path": path_match.group(1)}
                            }
                    elif call.startswith("web_search"):
                        query_match = re.search(r'query=[\'"](.+?)[\'"]', call)
                        if query_match:
                            data = {
                                "tool": "web_search",
                                "params": {"query": query_match.group(1)}
                            }

            # ---------- NORMAL JSON PARSE ----------
            if not data:
                try:
                    data = json.loads(out)
                except:
                    pass

            # ---------- FALLBACK JSON EXTRACTION ----------
            if not data:
                match = re.search(r'\{.*\}', out, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group())
                    except:
                        data = None

            if data and "tool" in data:
                tool = data["tool"]
                params = data.get("params", {})

                st.write(f"🛠 Tool: {tool}")
                st.json(params)

                if tool == "file_write":
                    if "content" not in params or not params["content"].strip():
                        st.error("file_write missing content")
                        return "ERROR", state

                res = tr.execute(tool, params, self.name)

                with st.expander("Tool Result", expanded=True):
                    st.code(str(res)[:1000])

                convo.append(f"Observation: {str(res)[:1000]}")

                if tool == "file_write":
                    state["last_file"] = params.get("path")
                    return f"File saved: {params.get('path')}", state

                continue

            st.success(f"{self.name} done")
            return out, state

        return "Max loops reached", state


# =============================================================================
# WORKFLOW
# =============================================================================
class Workflow:
    def __init__(self, agents):
        self.agents = agents

    def run(self, topic, mm, tr):
        state = {}
        # First agent gets the topic as context
        context = f"Topic: {topic}"
        st.header("Workflow Execution")

        for agent in self.agents:
            # Pass the current context and state
            context, state = agent.run(context, mm, tr, state)
            # Prepare context for next agent
            if agent.name == "Researcher":
                # After Researcher, pass topic + research results to Writer
                context = f"Topic: {topic}\nResearch results:\n{context}"
            elif agent.name == "Writer":
                # After Writer, the file path is in state
                file_path = state.get("last_file", "unknown")
                context = f"Topic: {topic}\nFile path: {file_path}"
            # Editor will use the above context

        st.success("Workflow Complete")
        return context, state


# =============================================================================
# STREAMLIT
# =============================================================================
def main():
    st.set_page_config(layout="wide")
    st.title("Research Generator")

    key = st.text_input("Gemini Key", type="password")

    if key:
        if "mm" not in st.session_state:
            st.session_state.mm = ModelManager(key)

        topic = st.text_input("Topic")

        if st.button("Run") and topic:
            tr = ToolRegistry()

            agents = [
                Agent(
                    "Researcher",
                    "Researcher",
                    """Search the given topic using web_search.

STRICT:
{
  "tool": "web_search",
  "params": {"query": "<topic>"}
}
""",
                    ["web_search"]
                ),
                Agent(
                    "Writer",
                    "Writer",
                    """You MUST write a complete article about the given topic and save it.

You will receive the topic and research results in the context.

STRICT RULES:
- Use the input topic only (ignore any other topics)
- Write a comprehensive, well‑structured article
- You MUST call file_write
- Output ONLY JSON

FORMAT:
{
  "tool": "file_write",
  "params": {
    "path": "<topic_based_filename>.md",
    "content": "FULL ARTICLE ABOUT THE INPUT TOPIC"
  }
}
""",
                    ["file_write"]
                ),
                Agent(
                    "Editor",
                    "Editor",
                    """You MUST:
1. Read the article using file_read (path provided in context)
2. Improve grammar, clarity, and structure
3. Save using file_write (overwrite the same file)
4. STOP after writing

STRICT:
- Output ONLY JSON
- Do not loop unnecessarily

Example:
{
  "tool": "file_read",
  "params": {"path": "article.md"}
}
After reading, output:
{
  "tool": "file_write",
  "params": {"path": "article.md", "content": "improved content"}
}
""",
                    ["file_read", "file_write"]
                )
            ]

            st.sidebar.markdown("## Agents")
            for a in agents:
                st.sidebar.write(a.name)

            wf = Workflow(agents)

            result, state = wf.run(topic, st.session_state.mm, tr)

            st.header("Final Output")
            st.write(result)

            if "last_file" in state:
                st.success(f"Saved: workspace/{state['last_file']}")


if __name__ == "__main__":
    init_db()
    main()