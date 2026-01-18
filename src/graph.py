"""
LangGraph orchestration for the Multi-Agent Debate System.

Defines the debate graph with:
- State schema with reducers (additive history)
- Node registration for all agents
- Conditional edges for phase routing
- Deterministic flow control
"""

import logging
from typing import Annotated, Any, Dict, List, Sequence
from typing_extensions import TypedDict
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .config import DebateConfig, get_default_config
from .models import DebateTurn, JudgeVerdict
from .agents import (
    create_proponent_node,
    create_opposition_node,
    create_judge_node,
    start_rebuttal_node,
    next_round_node,
    start_closing_node,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Graph State Definition
# ============================================================================

class GraphState(TypedDict):
    """
    LangGraph state schema for the debate.
    
    Uses Annotated types with operator.add for additive reducers,
    ensuring history is never overwritten, only appended.
    """
    # Configuration (set once at start)
    topic: str
    max_rounds: int
    
    # Phase tracking
    current_phase: str  # opening, rebuttal, closing, verdict, complete
    current_round: int
    
    # Debate history - uses ADD reducer for immutability
    history: Annotated[List[DebateTurn], operator.add]
    
    # Final verdict (set by judge)
    verdict: Dict[str, Any] | None
    
    # Error tracking
    errors: Annotated[List[str], operator.add]


# ============================================================================
# Router Functions
# ============================================================================

def route_after_proponent(state: GraphState) -> str:
    """
    Route after proponent speaks.
    
    Always goes to opposition for their response.
    """
    return "opposition"


def route_after_opposition(state: GraphState) -> str:
    """
    Route after opposition speaks.
    
    Determines next phase based on current state.
    """
    phase = state["current_phase"]
    current_round = state["current_round"]
    max_rounds = state["max_rounds"]
    
    if phase == "opening":
        # Opening complete, start rebuttals
        return "start_rebuttal"
    
    elif phase == "rebuttal":
        if current_round >= max_rounds:
            # All rebuttal rounds complete, start closing
            return "start_closing"
        else:
            # More rounds to go
            return "next_round"
    
    elif phase == "closing":
        # Closing complete, go to judge
        return "judge"
    
    # Fallback
    return END


def route_after_phase_change(state: GraphState) -> str:
    """
    Route after a phase transition node.
    
    Always starts with proponent.
    """
    return "proponent"


# ============================================================================
# Graph Construction
# ============================================================================

def create_debate_graph(config: DebateConfig | None = None) -> StateGraph:
    """
    Create and compile the debate graph.
    
    Args:
        config: Optional debate configuration (uses defaults if not provided)
        
    Returns:
        Compiled LangGraph StateGraph ready for execution
    
    Graph Structure:
        START
          │
          ▼
        proponent (opening)
          │
          ▼
        opposition (opening)
          │
          ▼
        start_rebuttal ───┐
          │               │
          ▼               │
        proponent ◄───────┤
          │               │
          ▼               │
        opposition        │
          │               │
          ├─(more rounds)─┘
          │
          ▼
        start_closing
          │
          ▼
        proponent (closing)
          │
          ▼
        opposition (closing)
          │
          ▼
        judge
          │
          ▼
        END
    """
    if config is None:
        config = get_default_config()
    
    logger.info(f"Creating debate graph with {config.max_rounds} max rounds")
    
    # Initialize graph with state schema
    graph = StateGraph(GraphState)
    
    # =========================================
    # Register Nodes
    # =========================================
    
    # Agent nodes
    graph.add_node("proponent", create_proponent_node(config))
    graph.add_node("opposition", create_opposition_node(config))
    graph.add_node("judge", create_judge_node(config))
    
    # Phase transition nodes
    graph.add_node("start_rebuttal", start_rebuttal_node)
    graph.add_node("next_round", next_round_node)
    graph.add_node("start_closing", start_closing_node)
    
    # =========================================
    # Define Edges
    # =========================================
    
    # Entry point
    graph.set_entry_point("proponent")
    
    # Proponent always goes to opposition
    graph.add_edge("proponent", "opposition")
    
    # Opposition routes based on phase
    graph.add_conditional_edges(
        "opposition",
        route_after_opposition,
        {
            "start_rebuttal": "start_rebuttal",
            "start_closing": "start_closing",
            "next_round": "next_round",
            "judge": "judge",
            END: END,
        }
    )
    
    # Phase transitions go back to proponent
    graph.add_edge("start_rebuttal", "proponent")
    graph.add_edge("next_round", "proponent")
    graph.add_edge("start_closing", "proponent")
    
    # Judge ends the graph
    graph.add_edge("judge", END)
    
    # =========================================
    # Compile and Return
    # =========================================
    
    # Optional: Add memory saver for checkpointing
    # memory = MemorySaver()
    # return graph.compile(checkpointer=memory)
    
    return graph.compile()


# ============================================================================
# Convenience Functions
# ============================================================================

def run_debate(
    topic: str,
    max_rounds: int = 3,
    config: DebateConfig | None = None,
) -> Dict[str, Any]:
    """
    Run a complete debate on the given topic.
    
    Args:
        topic: The debate proposition
        max_rounds: Number of rebuttal rounds
        config: Optional configuration override
        
    Returns:
        Final state dict with complete debate history and verdict
    """
    if config is None:
        config = get_default_config()
    
    # Override max_rounds if specified
    if max_rounds != config.max_rounds:
        config = DebateConfig(
            **{**config.model_dump(), "max_rounds": max_rounds}
        )
    
    # Create graph
    graph = create_debate_graph(config)
    
    # Initial state
    initial_state: GraphState = {
        "topic": topic,
        "max_rounds": config.max_rounds,
        "current_phase": "opening",
        "current_round": 0,
        "history": [],
        "verdict": None,
        "errors": [],
    }
    
    logger.info(f"Starting debate: '{topic}'")
    logger.info(f"Configuration: {config.max_rounds} rounds, model: {config.model_name}")
    
    # Run the graph
    final_state = graph.invoke(initial_state)
    
    logger.info("Debate complete")
    
    return final_state


def stream_debate(
    topic: str,
    max_rounds: int = 3,
    config: DebateConfig | None = None,
):
    """
    Stream debate execution, yielding each turn as it completes.
    
    Args:
        topic: The debate proposition
        max_rounds: Number of rebuttal rounds
        config: Optional configuration override
        
    Yields:
        State updates after each node execution
    """
    if config is None:
        config = get_default_config()
    
    if max_rounds != config.max_rounds:
        config = DebateConfig(
            **{**config.model_dump(), "max_rounds": max_rounds}
        )
    
    graph = create_debate_graph(config)
    
    initial_state: GraphState = {
        "topic": topic,
        "max_rounds": config.max_rounds,
        "current_phase": "opening",
        "current_round": 0,
        "history": [],
        "verdict": None,
        "errors": [],
    }
    
    logger.info(f"Starting streaming debate: '{topic}'")
    
    for event in graph.stream(initial_state, stream_mode="updates"):
        yield event
