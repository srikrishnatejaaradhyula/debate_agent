"""
Multi-Agent Debate System

A production-grade multi-agent debate system using LangChain and LangGraph
where multiple LLM agents debate a topic and produce a structured verdict.
"""

from .config import DebateConfig
from .models import DebateState, DebateTurn, JudgeVerdict
from .graph import create_debate_graph

__all__ = [
    "DebateConfig",
    "DebateState",
    "DebateTurn", 
    "JudgeVerdict",
    "create_debate_graph",
]
