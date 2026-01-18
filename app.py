"""
Streamlit Chat-Based UI for Multi-Agent Debate System

A real-time debate viewer that displays agent turns incrementally
as they unfold, with proper session state management.

Architecture:
- Chat-based interface using st.chat_message()
- Session state preserves debate history across reruns
- Streaming support renders messages as agents respond
- Sidebar controls for rounds, model selection, and reset

Run:
    streamlit run app.py
"""

import streamlit as st
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

# Import debate system components
from src.config import DebateConfig
from src.graph import stream_debate, run_debate
from src.models import DebateTurn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="AI Debate Arena",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Custom CSS for Enhanced UI
# ============================================================================

st.markdown("""
<style>
    /* Agent message styling */
    .proponent-msg {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .opposition-msg {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .judge-msg {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Verdict panel */
    .verdict-panel {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border: 2px solid #ff9800;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Phase badge */
    .phase-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: bold;
        text-transform: uppercase;
        margin-left: 0.5rem;
    }
    
    .phase-opening { background: #e1bee7; color: #7b1fa2; }
    .phase-rebuttal { background: #b3e5fc; color: #0277bd; }
    .phase-closing { background: #c5cae9; color: #303f9f; }
    .phase-verdict { background: #ffcc80; color: #e65100; }
    
    /* Thinking indicator */
    .thinking-box {
        background: linear-gradient(90deg, #f5f5f5, #e0e0e0, #f5f5f5);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Better message container */
    .stChatMessage {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state():
    """
    Initialize session state variables.
    
    Called at app startup to ensure all required state exists.
    """
    if "debate_messages" not in st.session_state:
        st.session_state.debate_messages = []
    
    if "current_round" not in st.session_state:
        st.session_state.current_round = 0
    
    if "final_verdict" not in st.session_state:
        st.session_state.final_verdict = None
    
    if "debate_in_progress" not in st.session_state:
        st.session_state.debate_in_progress = False
    
    if "debate_topic" not in st.session_state:
        st.session_state.debate_topic = None
    
    if "max_rounds" not in st.session_state:
        st.session_state.max_rounds = 2
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "deepseek/deepseek-r1-0528:free"
    
    if "pending_topic" not in st.session_state:
        st.session_state.pending_topic = None


def reset_debate():
    """Reset debate state for a new debate."""
    st.session_state.debate_messages = []
    st.session_state.current_round = 0
    st.session_state.final_verdict = None
    st.session_state.debate_in_progress = False
    st.session_state.debate_topic = None
    st.session_state.pending_topic = None


# ============================================================================
# Agent Message Rendering
# ============================================================================

# Agent configuration with icons and colors
AGENT_CONFIG = {
    "proponent": {
        "icon": "🟢",
        "name": "Proponent",
        "avatar": "🟢",
        "css_class": "proponent-msg",
        "thinking_text": "🟢 Proponent is thinking...",
    },
    "opposition": {
        "icon": "🔴", 
        "name": "Opposition",
        "avatar": "🔴",
        "css_class": "opposition-msg",
        "thinking_text": "🔴 Opposition is thinking...",
    },
    "judge": {
        "icon": "⚖️",
        "name": "Judge",
        "avatar": "⚖️",
        "css_class": "judge-msg",
        "thinking_text": "⚖️ Judge is deliberating...",
    },
}

PHASE_BADGES = {
    "opening": ("🎬 Opening", "phase-opening"),
    "rebuttal": ("⚔️ Rebuttal", "phase-rebuttal"),
    "closing": ("🎯 Closing", "phase-closing"),
    "verdict": ("⚖️ Verdict", "phase-verdict"),
}


def render_agent_message(turn: Dict[str, Any], index: int):
    """
    Render a single agent message in chat format.
    
    Args:
        turn: Dictionary with role, phase, round_number, content
        index: Message index for unique keys
    """
    role = turn.get("role", "unknown")
    phase = turn.get("phase", "unknown")
    round_num = turn.get("round_number", 0)
    content = turn.get("content", "")
    word_count = turn.get("word_count", len(content.split()))
    
    config = AGENT_CONFIG.get(role, AGENT_CONFIG["proponent"])
    phase_label, phase_class = PHASE_BADGES.get(phase, ("Unknown", ""))
    
    # Build header
    header = f"{config['icon']} **{config['name']}**"
    if phase == "rebuttal" and round_num > 0:
        header += f" • Round {round_num}"
    
    # Render using Streamlit chat message
    with st.chat_message(role, avatar=config["avatar"]):
        # Header with phase badge
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {header}")
        with col2:
            st.markdown(f"<span class='phase-badge {phase_class}'>{phase_label}</span>", 
                       unsafe_allow_html=True)
        
        # Content with expander for long messages
        if word_count > 200:
            with st.expander(f"View full response ({word_count} words)", expanded=True):
                st.markdown(content)
        else:
            st.markdown(content)
        
        # Footer with metadata
        st.caption(f"📝 {word_count} words")


def render_thinking_indicator(role: str, placeholder):
    """
    Render a thinking indicator for an agent.
    
    Args:
        role: Agent role (proponent, opposition, judge)
        placeholder: Streamlit placeholder to render in
    """
    config = AGENT_CONFIG.get(role, AGENT_CONFIG["proponent"])
    
    with placeholder.container():
        with st.chat_message(role, avatar=config["avatar"]):
            st.markdown(f"### {config['thinking_text']}")
            st.markdown("""
            <div class="thinking-box">
                <em>Formulating arguments...</em>
            </div>
            """, unsafe_allow_html=True)


def render_verdict_panel(verdict: Dict[str, Any]):
    """
    Render the final judge verdict in a highlighted panel.
    
    Args:
        verdict: Dictionary containing verdict information
    """
    st.markdown("---")
    st.markdown("## ⚖️ Final Verdict")
    
    # Winner announcement
    winner = verdict.get("winner", "Unknown")
    confidence = verdict.get("confidence", "medium")
    
    winner_emoji = "🟢" if winner == "proponent" else "🔴" if winner == "opposition" else "⚪"
    confidence_emoji = "🔥" if confidence == "high" else "💡" if confidence == "medium" else "❓"
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Winner",
            value=f"{winner_emoji} {winner.upper()}",
            delta=None,
        )
    with col2:
        st.metric(
            label="Confidence",
            value=f"{confidence_emoji} {confidence.upper()}",
            delta=None,
        )
    
    # Summary
    summary = verdict.get("summary", "")
    if summary:
        st.info(f"📋 {summary}")
    
    # Detailed reasoning
    reasoning = verdict.get("reasoning", "")
    if reasoning:
        with st.expander("📖 Detailed Analysis", expanded=False):
            st.markdown(reasoning)


# ============================================================================
# Debate Execution (Using Placeholders - No Rerun)
# ============================================================================

def run_debate_with_ui(topic: str, max_rounds: int, model: str, message_container):
    """
    Execute a debate, rendering each message as it completes.
    
    Uses placeholders to update UI without triggering reruns.
    
    Args:
        topic: The debate proposition
        max_rounds: Number of rebuttal rounds
        model: OpenRouter model to use
        message_container: Streamlit container to render messages in
    """
    # Create config with selected model
    try:
        config = DebateConfig(
            model_name=model,
            max_rounds=max_rounds,
        )
    except ValueError as e:
        st.error(f"Configuration error: {e}")
        return
    
    # Track which agents we expect next
    debate_flow = []
    
    # Opening
    debate_flow.append(("proponent", "opening", 0))
    debate_flow.append(("opposition", "opening", 0))
    
    # Rebuttals
    for r in range(1, max_rounds + 1):
        debate_flow.append(("proponent", "rebuttal", r))
        debate_flow.append(("opposition", "rebuttal", r))
    
    # Closing
    debate_flow.append(("proponent", "closing", 0))
    debate_flow.append(("opposition", "closing", 0))
    
    # Judge
    debate_flow.append(("judge", "verdict", 0))
    
    try:
        # Use streaming to get messages one by one
        message_index = 0
        
        for event in stream_debate(topic=topic, max_rounds=max_rounds, config=config):
            for node_name, node_output in event.items():
                # Handle new turns
                new_turns = node_output.get("history", [])
                for turn in new_turns:
                    # Convert DebateTurn to dict for storage
                    turn_dict = {
                        "role": turn.role,
                        "phase": turn.phase,
                        "round_number": turn.round_number,
                        "content": turn.content,
                        "word_count": turn.word_count,
                        "timestamp": turn.timestamp.isoformat(),
                    }
                    
                    # Store in session state
                    st.session_state.debate_messages.append(turn_dict)
                    
                    # Update current round
                    if turn.round_number > st.session_state.current_round:
                        st.session_state.current_round = turn.round_number
                    
                    # Render the message immediately
                    with message_container:
                        render_agent_message(turn_dict, message_index)
                    
                    message_index += 1
                    
                    # Show who's thinking next (if not judge verdict)
                    if message_index < len(debate_flow):
                        next_role, next_phase, _ = debate_flow[message_index]
                        config = AGENT_CONFIG.get(next_role, AGENT_CONFIG["proponent"])
                        with message_container:
                            with st.status(config["thinking_text"], expanded=True) as status:
                                st.write(f"Preparing {next_phase} arguments...")
                
                # Handle verdict
                if "verdict" in node_output and node_output["verdict"]:
                    st.session_state.final_verdict = node_output["verdict"]
                    
    except Exception as e:
        st.error(f"Debate error: {e}")
        logger.exception("Debate failed")
    
    finally:
        st.session_state.debate_in_progress = False


# ============================================================================
# Export Functions
# ============================================================================

def export_transcript_markdown() -> str:
    """Export debate transcript as Markdown."""
    lines = [
        f"# Debate Transcript",
        f"**Topic:** {st.session_state.debate_topic}",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Rounds:** {st.session_state.max_rounds}",
        "",
        "---",
        "",
    ]
    
    for msg in st.session_state.debate_messages:
        role = msg["role"].upper()
        phase = msg["phase"].upper()
        round_num = msg.get("round_number", 0)
        
        header = f"## {AGENT_CONFIG[msg['role']]['icon']} {role} - {phase}"
        if phase == "REBUTTAL" and round_num > 0:
            header += f" (Round {round_num})"
        
        lines.append(header)
        lines.append("")
        lines.append(msg["content"])
        lines.append("")
        lines.append("---")
        lines.append("")
    
    if st.session_state.final_verdict:
        verdict = st.session_state.final_verdict
        lines.append("## ⚖️ FINAL VERDICT")
        lines.append("")
        lines.append(f"**Winner:** {verdict['winner'].upper()}")
        lines.append(f"**Confidence:** {verdict['confidence'].upper()}")
        lines.append("")
        lines.append(verdict.get("summary", ""))
    
    return "\n".join(lines)


def export_transcript_json() -> str:
    """Export debate transcript as JSON."""
    data = {
        "topic": st.session_state.debate_topic,
        "timestamp": datetime.now().isoformat(),
        "max_rounds": st.session_state.max_rounds,
        "messages": st.session_state.debate_messages,
        "verdict": st.session_state.final_verdict,
    }
    return json.dumps(data, indent=2)


# ============================================================================
# Sidebar UI
# ============================================================================

def render_sidebar():
    """Render the sidebar with controls and settings."""
    with st.sidebar:
        st.title("⚙️ Debate Settings")
        
        # Model selection
        st.subheader("🤖 Model")
        
        # Note: Free models require OpenRouter privacy settings enabled
        # https://openrouter.ai/settings/privacy
        models = [
            "deepseek/deepseek-r1-0528:free",
            "deepseek/deepseek-chat-v3-0324:free",
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "qwen/qwen3-235b-a22b:free",
        ]
        st.session_state.selected_model = st.selectbox(
            "Select Model",
            options=models,
            index=0,
            disabled=st.session_state.debate_in_progress,
            help="Free models require privacy settings enabled at openrouter.ai/settings/privacy"
        )
        
        # Rounds selection
        st.subheader("🔄 Rounds")
        st.session_state.max_rounds = st.slider(
            "Rebuttal Rounds",
            min_value=1,
            max_value=5,
            value=st.session_state.max_rounds,
            disabled=st.session_state.debate_in_progress,
        )
        
        st.markdown("---")
        
        # Reset button
        if st.button(
            "🔄 Reset Debate",
            disabled=st.session_state.debate_in_progress,
            use_container_width=True,
        ):
            reset_debate()
            st.rerun()
        
        # Export options (only if debate exists)
        if st.session_state.debate_messages:
            st.markdown("---")
            st.subheader("📥 Export")
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📄 Markdown",
                    data=export_transcript_markdown(),
                    file_name="debate_transcript.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col2:
                st.download_button(
                    "📋 JSON",
                    data=export_transcript_json(),
                    file_name="debate_transcript.json",
                    mime="application/json",
                    use_container_width=True,
                )
        
        # Debate status
        st.markdown("---")
        if st.session_state.debate_in_progress:
            st.warning("🔄 Debate in progress...")
        elif st.session_state.debate_messages:
            st.success(f"✅ Debate complete ({len(st.session_state.debate_messages)} turns)")
        else:
            st.info("💬 Enter a topic to start")


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main Streamlit application entry point."""
    # Initialize session state
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main header
    st.title("⚖️ AI Debate Arena")
    st.markdown("*Watch AI agents debate any topic in real-time*")
    
    # Show current topic if in debate
    if st.session_state.debate_topic:
        st.markdown(f"### 📋 Topic: *{st.session_state.debate_topic}*")
        st.markdown("---")
    
    # Create container for debate messages
    message_container = st.container()
    
    # Render existing messages
    with message_container:
        for i, msg in enumerate(st.session_state.debate_messages):
            render_agent_message(msg, i)
    
    # Render verdict if available
    if st.session_state.final_verdict:
        render_verdict_panel(st.session_state.final_verdict)
    
    # Handle pending topic (set from previous run)
    if st.session_state.pending_topic and not st.session_state.debate_in_progress:
        topic = st.session_state.pending_topic
        st.session_state.pending_topic = None
        st.session_state.debate_topic = topic
        st.session_state.debate_in_progress = True
        
        # Show thinking indicator for first agent
        with message_container:
            with st.status("🟢 Proponent is thinking...", expanded=True) as status:
                st.write("Preparing opening arguments...")
                
                # Run the debate
                run_debate_with_ui(
                    topic=topic,
                    max_rounds=st.session_state.max_rounds,
                    model=st.session_state.selected_model,
                    message_container=message_container
                )
                
                status.update(label="✅ Debate complete!", state="complete", expanded=False)
        
        # Show verdict
        if st.session_state.final_verdict:
            render_verdict_panel(st.session_state.final_verdict)
    
    # Chat input for new debate
    if not st.session_state.debate_in_progress and not st.session_state.debate_messages:
        topic = st.chat_input(
            "Enter a debate topic...",
            disabled=st.session_state.debate_in_progress,
        )
        
        if topic:
            # Show user message
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"**Debate Topic:** {topic}")
            
            # Set pending topic and rerun to start debate
            st.session_state.pending_topic = topic
            st.rerun()
    
    # If debate complete, show input for new debate
    elif not st.session_state.debate_in_progress and st.session_state.debate_messages:
        st.markdown("---")
        st.info("💡 Click **Reset Debate** in the sidebar to start a new debate.")


if __name__ == "__main__":
    main()
