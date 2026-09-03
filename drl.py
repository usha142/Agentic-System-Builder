# OpenRouter version
# Install dependencies:
# pip install -U streamlit langchain-openrouter langchain-core langgraph pandas
#
# OpenRouter API keys normally start with: sk-or-v1-...
# The app uses OpenRouter's OpenAI-compatible endpoint:
# https://openrouter.ai/api/v1
#
# ==========================================
# DYNAMIC AGENTIC SYSTEM BUILDER
# ==========================================
import streamlit as st
import os
import sys
import subprocess
import json
import pandas as pd
import glob
from typing import TypedDict, List, Optional, Dict, Any
from datetime import datetime

# LangChain / LangGraph Imports
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="Nexus: Agentic System Builder",
    page_icon="🧬",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .stChatMessage { border: 1px solid #e0e0e0; border-radius: 10px; padding: 10px; }
    .stCode { font-family: 'Fira Code', monospace; }
    .agent-card { background-color: #f0f2f6; padding: 10px; border-radius: 8px; margin-bottom: 5px; border-left: 4px solid #4CAF50; }
    .success { color: #155724; background-color: #d4edda; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- 2. PERSISTENCE & WORKSPACE ---
WORKSPACE_DIR = "workspace"
DATA_FILE = "agent_systems.json"
os.makedirs(WORKSPACE_DIR, exist_ok=True)

def load_data():
    """Loads systems and history from JSON."""
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_data(data):
    """Saves systems to JSON."""
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# --- 3. TOOL: PYTHON SANDBOX ---
def execute_python_code(code: str):
    """Executes Python code in a local environment."""
    file_path = os.path.join(WORKSPACE_DIR, "script.py")
    clean_code = code.replace("```python", "").replace("```", "").strip()
    
    with open(file_path, "w", encoding='utf-8') as f:
        f.write(clean_code)
    
    try:
        result = subprocess.run(
            [sys.executable, "script.py"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.abspath(WORKSPACE_DIR) 
        )
        if result.returncode != 0:
            return False, f"RUNTIME ERROR:\n{result.stderr}"
        return True, f"SUCCESS:\n{result.stdout}"
    except Exception as e:
        return False, f"SYSTEM ERROR: {str(e)}"

# --- 4. DYNAMIC GRAPH BUILDER ---

# State Definition
class AgentState(TypedDict):
    messages: List[Any]         # Chat History
    next_agent: str             # Who acts next
    current_code: Optional[str] # Code to execute
    tool_output: Optional[str]  # Result of code execution
    sender: str                 # Last active agent

def get_llm(api_key, model="openrouter/free"):
    """Create a LangChain ChatOpenRouter client."""
    return ChatOpenRouter(
        model=model,
        temperature=0,
        api_key=api_key,
        max_retries=2,
        max_tokens=2000,
    )


# Node: The Dynamic Supervisor
def supervisor_node(state, config):
    system_config = config['configurable']['system_config']
    api_key = config['configurable']['api_key']
    llm = get_llm(api_key, config['configurable'].get('model', 'openrouter/free'))

    agent_names = [
        str(a.get('Role', '')).strip()
        for a in system_config.get('agents', [])
        if str(a.get('Role', '')).strip()
    ]
    agent_list_str = ", ".join(agent_names)

    if not agent_names:
        return {"next_agent": "FINISH", "sender": "Supervisor"}

    system_prompt = (
        f"You are the Supervisor. You manage these agents: {agent_list_str}.\\n"
        "Your job is to route the conversation to the best agent.\\n"
        "1. If the user asks a question, pick the most relevant expert.\\n"
        "2. If an agent has just finished and the task is completely done, reply FINISH.\\n"
        "3. If an agent needs help from another, route to that agent.\\n"
        f"Return exactly ONE of these values and nothing else: {agent_list_str}, FINISH. "
        "Do not explain your choice. Do not provide reasoning. Do not write a plan."
    )

    messages = state['messages']

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"),
    ])
    chain = prompt | llm
    result = chain.invoke({"messages": messages})

    raw_result = result.content if hasattr(result, "content") else str(result)
    raw_result = str(raw_result).strip()

    # OpenRouter models can ignore "ONLY return..." and emit reasoning.
    # Extract a valid configured agent name from that response.
    next_step = None
    for agent_name in agent_names:
        if agent_name.lower() in raw_result.lower():
            next_step = agent_name
            break

    if next_step is None and "FINISH" in raw_result.upper():
        next_step = "FINISH"

    # Never terminate just because the model returned unexpected formatting.
    if next_step is None:
        next_step = agent_names[0]

    return {"next_agent": next_step, "sender": "Supervisor"}


# Node: Generic Worker (Factory)
def create_worker_node(agent_name, agent_instructions):
    def worker_node(state, config):
        api_key = config['configurable']['api_key']
        llm = get_llm(api_key, config['configurable'].get('model', 'openrouter/free'))
        
        # Add Tool instructions if not present
        tool_instr = "\nIf you need to calculate or process data, write Python code inside ```python``` blocks. The system will execute it."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", agent_instructions + tool_instr),
            ("placeholder", "{messages}"),
        ])
        chain = prompt | llm
        response = chain.invoke({"messages": state['messages']})
        
        # Check for code
        code_block = None
        if "```python" in response.content:
            try:
                code_block = response.content.split("```python")[1].split("```")[0]
            except:
                pass
                
        return {
            "messages": [response], 
            "sender": agent_name, 
            "current_code": code_block
        }
    return worker_node

# Node: Tool Executor
def tool_node(state, config):
    code = state['current_code']
    success, output = execute_python_code(code)
    result_msg = f"Tool Execution Result:\n{output}"
    return {
        "messages": [HumanMessage(content=result_msg, name="Tool_Executor")],
        "sender": "Tool_Executor",
        "current_code": None # Reset
    }

# Logic: Build the Graph dynamically
def build_dynamic_graph(system_config):
    workflow = StateGraph(AgentState)
    
    # 1. Add Supervisor
    workflow.add_node("Supervisor", supervisor_node)
    
    # 2. Add Workers from Config
    agent_names = []
    for agent in system_config['agents']:
        name = agent['Role']
        instr = agent['Instructions']
        workflow.add_node(name, create_worker_node(name, instr))
        agent_names.append(name)
        
    # 3. Add Tool Node
    workflow.add_node("Tool_Executor", tool_node)
    
    # 4. Edges
    workflow.set_entry_point("Supervisor")
    
    # Supervisor Routing Logic
    def route_supervisor(state):
        return state['next_agent']
    
    mapping = {name: name for name in agent_names}
    mapping["FINISH"] = END
    workflow.add_conditional_edges("Supervisor", route_supervisor, mapping)
    
    # Worker Routing Logic (Check for code execution)
    def route_worker(state):
        if state.get('current_code'):
            return "tools"
        return "supervisor"
        
    for name in agent_names:
        workflow.add_conditional_edges(name, route_worker, {"tools": "Tool_Executor", "supervisor": "Supervisor"})
        
    # Tool always goes back to Supervisor to decide next steps (or back to worker)
    # For simplicity, let's send tool output back to Supervisor to re-route
    workflow.add_edge("Tool_Executor", "Supervisor")
    
    return workflow.compile()

# --- 5. UI & APP LOGIC ---

# Sidebar
with st.sidebar:
    st.header("🧬 Agentic System Builder")
    api_key = st.text_input("OpenRouter API Key", type="password")
    st.caption("💡 For testing without paid credits, use `openrouter/free`.")
    model_name = st.text_input(
        "OpenRouter Model",
        value="openrouter/free",
        help="Use an OpenRouter model slug, e.g. openrouter/free or a specific provider/model slug."
    )
    
    # Load Data
    all_systems = load_data()
    system_names = list(all_systems.keys())
    
    # System Selector
    selected_system_name = st.selectbox("Select Active System", ["-- Create New --"] + system_names)
    
    # Create New System Interface
    if selected_system_name == "-- Create New --":
        st.divider()
        st.subheader("🛠️ Create New System")
        new_sys_name = st.text_input("System Name", placeholder="e.g. Market Research Team")
        uploaded_csv = st.file_uploader("Upload Instructions (CSV)", type=["csv"], help="Columns: Role, Instructions")
        
        if st.button("Create System"):
            if new_sys_name and uploaded_csv:
                try:
                    df = pd.read_csv(uploaded_csv)
                    if "Role" in df.columns and "Instructions" in df.columns:
                        # Prevent blank CSV cells from becoming float NaN values.
                        df["Role"] = df["Role"].fillna("").astype(str)
                        df["Instructions"] = df["Instructions"].fillna("").astype(str)

                        agents_config = df.to_dict(orient="records")
                        all_systems[new_sys_name] = {
                            "created_at": str(datetime.now()),
                            "agents": agents_config,
                            "history": []
                        }
                        save_data(all_systems)
                        st.success(f"System '{new_sys_name}' created! Select it from the dropdown.")
                        st.rerun()
                    else:
                        st.error("CSV must have 'Role' and 'Instructions' columns.")
                except Exception as e:
                    st.error(f"Error parsing CSV: {e}")
            else:
                st.warning("Please provide a name and a CSV file.")

# Main Area
if selected_system_name != "-- Create New --" and selected_system_name in all_systems:
    active_system = all_systems[selected_system_name]
    
    st.title(f"🤖 {selected_system_name}")
    
    # Show Active Agents
    with st.expander("View Agent Team Configuration"):
        cols = st.columns(len(active_system['agents']))
        for idx, agent in enumerate(active_system['agents']):
            with cols[idx % 3]:
                st.markdown(f"**{agent.get('Role', 'Unnamed Agent')}**")

                # CSV values can contain NaN (float) when Instructions is blank.
                # Convert safely to text before slicing.
                instructions = agent.get('Instructions', '')
                if pd.isna(instructions):
                    instructions = ''
                else:
                    instructions = str(instructions)

                preview = instructions[:1000]
                st.caption(preview + ("..." if len(instructions) > 1000 else ""))

    # Chat Interface
    chat_container = st.container()
    
    # Load History
    if "messages" not in st.session_state:
        st.session_state.messages = active_system.get("history", [])

    with chat_container:
        for msg in st.session_state.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            name = msg.get("name", "")
            
            with st.chat_message(role):
                if name: st.markdown(f"**{name}:**")
                st.markdown(content)

    # Input
    user_input = st.chat_input("Enter your request...")
    
    if user_input:
        if not api_key:
            st.error("Please enter your OpenRouter API Key.")
        else:
            # Add User Message
            user_msg_obj = {"role": "user", "content": user_input}
            st.session_state.messages.append(user_msg_obj)
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Build Graph
            app = build_dynamic_graph(active_system)
            
            # Prepare LangChain messages for the graph
            # We convert our JSON history dicts back to LangChain objects
            lc_messages = []
            for m in st.session_state.messages:
                if m['role'] == 'user':
                    lc_messages.append(HumanMessage(content=m['content']))
                elif m['role'] == 'assistant':
                    lc_messages.append(AIMessage(content=m['content']))
            
            initial_state = {
                "messages": lc_messages,
                "next_agent": "Supervisor",
                "current_code": None
            }
            
            config = {"configurable": {"api_key": api_key, "system_config": active_system, "model": model_name}}
            
            # Stream execution
            with st.chat_message("assistant"):
                status_box = st.status("⚙️ Orchestrating Agents...", expanded=True)
                
                try:
                    max_steps = 12
                    step_count = 0

                    for event in app.stream(initial_state, config=config):
                        step_count += 1
                        if step_count > max_steps:
                            raise RuntimeError(
                                f"Agent workflow exceeded {max_steps} steps. "
                                "Stopped to prevent an accidental agent loop."
                            )

                        for node_name, node_state in event.items():
                            sender = node_state.get('sender', 'System')
                            
                            # Show intermediate steps in status
                            if node_name == "Supervisor":
                                next_act = node_state.get('next_agent')
                                status_box.write(f"👑 **Supervisor:** Delegating to -> {next_act}")
                                
                            elif node_name == "Tool_Executor":
                                status_box.write("🛠️ **Tool:** Code Executed.")
                                msgs = node_state['messages']
                                if msgs:
                                    last_msg = msgs[-1].content
                                    # Append tool output to chat history visually
                                    st.code(last_msg[:500] + "..." if len(last_msg)>500 else last_msg)
                                    # Add to session
                                    st.session_state.messages.append({"role": "assistant", "name": "Tool", "content": f"```\n{last_msg}\n```"})

                            else:
                                # Worker Agent Response
                                msgs = node_state['messages']
                                if msgs:
                                    last_msg = msgs[-1].content
                                    status_box.write(f"👤 **{sender}:** responded.")
                                    st.markdown(f"**{sender}**: {last_msg}")
                                    
                                    # Add to session
                                    st.session_state.messages.append({"role": "assistant", "name": sender, "content": last_msg})
                                    
                    status_box.update(label="Task Complete", state="complete", expanded=False)
                    
                    # Persist History
                    active_system['history'] = st.session_state.messages
                    all_systems[selected_system_name] = active_system
                    save_data(all_systems)
                    
                except Exception as e:
                    status_box.update(label="Error", state="error")
                    error_text = str(e)

                    st.error(f"Execution Error: {error_text}")

                    with st.expander("🔍 Troubleshooting"):
                        st.markdown("""
                        **Check these first:**
                        1. Your key should be an OpenRouter key beginning with `sk-or-`.
                        2. Make sure the OpenRouter account has available credits or access to the selected free model.
                        3. Verify the model is a valid OpenRouter `provider/model` slug.
                        4. Make sure your computer can reach `https://openrouter.ai`.
                        5. If you are behind a college/company proxy or firewall, it may be blocking the API connection.

                        The current default model is `openrouter/free`, with a 2,000-token output cap.
                        """)

elif selected_system_name == "-- Create New --":
    st.info("👈 Use the sidebar to create your first Agentic System.")