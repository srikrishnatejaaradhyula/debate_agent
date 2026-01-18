"""
Agent node implementations for the Multi-Agent Debate System.

Each node:
1. Retrieves state from the graph
2. Constructs context-aware prompt
3. Invokes LLM with retry logic
4. Validates output format
5. Returns additive state update

Nodes follow LangGraph conventions: (state) -> partial state dict
"""

import logging
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .config import DebateConfig
from .models import DebateState, DebateTurn
from .prompts import (
    build_proponent_prompt,
    build_opposition_prompt,
    build_judge_prompt,
)
from .utils import (
    retry_with_backoff,
    validate_proponent_output,
    validate_opposition_output,
    validate_judge_output,
    truncate_response,
    clean_response,
    extract_winner_from_text,
    extract_confidence_from_text,
)

logger = logging.getLogger(__name__)


# ============================================================================
# LLM Client Factory
# ============================================================================

def create_llm_client(config: DebateConfig) -> ChatOpenAI:
    """
    Create a configured LLM client for OpenRouter.
    
    Args:
        config: Debate configuration with API settings
        
    Returns:
        Configured ChatOpenAI instance
    """
    return ChatOpenAI(
        model=config.model_name,
        openai_api_key=config.openrouter_api_key,
        openai_api_base=config.openrouter_base_url,
        temperature=config.temperature,
        max_tokens=2000,  # Reasonable limit for debate responses
        default_headers={
            "HTTP-Referer": "https://debate-agent.local",
            "X-Title": "Multi-Agent Debate System",
        }
    )


# ============================================================================
# Agent Invocation Helper
# ============================================================================

def invoke_agent(
    llm: ChatOpenAI,
    prompt: str,
    config: DebateConfig
) -> str:
    """
    Invoke LLM with retry logic.
    
    Args:
        llm: Configured LLM client
        prompt: Complete prompt string
        config: Configuration for retry settings
        
    Returns:
        LLM response content
    """
    @retry_with_backoff(
        max_retries=config.max_retries,
        base_delay=config.retry_delay,
    )
    def _invoke():
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    return _invoke()


# ============================================================================
# Proponent Agent Node
# ============================================================================

def create_proponent_node(config: DebateConfig):
    """
    Factory function to create a proponent agent node.
    
    Args:
        config: Debate configuration
        
    Returns:
        Node function for LangGraph
    """
    llm = create_llm_client(config)
    
    def proponent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute proponent agent turn.
        
        Constructs prompt based on current phase and history,
        invokes LLM, validates output, and returns state update.
        """
        logger.info(f"Proponent agent executing - Phase: {state['current_phase']}, Round: {state['current_round']}")
        
        # Build context-aware prompt
        prompt = build_proponent_prompt(
            topic=state["topic"],
            phase=state["current_phase"],
            round_number=state["current_round"],
            history=state["history"],
            max_words=config.max_response_length,
        )
        
        # Invoke LLM
        raw_response = invoke_agent(llm, prompt, config)
        
        # Clean and truncate response
        response = clean_response(raw_response)
        response = truncate_response(response, config.max_response_length)
        
        # Validate output format (log warning if invalid, but continue)
        is_valid, missing = validate_proponent_output(response)
        if not is_valid:
            logger.warning(f"Proponent output missing headers: {missing}")
        
        # Create turn record
        turn = DebateTurn(
            role="proponent",
            phase=state["current_phase"],
            round_number=state["current_round"],
            content=response,
        )
        
        logger.info(f"Proponent turn complete - {turn.word_count} words")
        
        # Return additive state update
        return {
            "history": [turn],  # Will be added via reducer
        }
    
    return proponent_node


# ============================================================================
# Opposition Agent Node
# ============================================================================

def create_opposition_node(config: DebateConfig):
    """
    Factory function to create an opposition agent node.
    
    Args:
        config: Debate configuration
        
    Returns:
        Node function for LangGraph
    """
    llm = create_llm_client(config)
    
    def opposition_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute opposition agent turn.
        
        Responds to proponent's arguments with counterarguments.
        """
        logger.info(f"Opposition agent executing - Phase: {state['current_phase']}, Round: {state['current_round']}")
        
        # Build context-aware prompt
        prompt = build_opposition_prompt(
            topic=state["topic"],
            phase=state["current_phase"],
            round_number=state["current_round"],
            history=state["history"],
            max_words=config.max_response_length,
        )
        
        # Invoke LLM
        raw_response = invoke_agent(llm, prompt, config)
        
        # Clean and truncate response
        response = clean_response(raw_response)
        response = truncate_response(response, config.max_response_length)
        
        # Validate output format
        is_valid, missing = validate_opposition_output(response)
        if not is_valid:
            logger.warning(f"Opposition output missing headers: {missing}")
        
        # Create turn record
        turn = DebateTurn(
            role="opposition",
            phase=state["current_phase"],
            round_number=state["current_round"],
            content=response,
        )
        
        logger.info(f"Opposition turn complete - {turn.word_count} words")
        
        # Return additive state update
        return {
            "history": [turn],
        }
    
    return opposition_node


# ============================================================================
# Judge Agent Node
# ============================================================================

def create_judge_node(config: DebateConfig):
    """
    Factory function to create a judge agent node.
    
    Args:
        config: Debate configuration
        
    Returns:
        Node function for LangGraph
    """
    llm = create_llm_client(config)
    
    def judge_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute judge agent to produce final verdict.
        
        Analyzes complete debate history and produces structured verdict.
        """
        logger.info("Judge agent executing - Producing final verdict")
        
        # Build complete debate context
        prompt = build_judge_prompt(
            topic=state["topic"],
            history=state["history"],
        )
        
        # Invoke LLM (judge may need more tokens)
        raw_response = invoke_agent(llm, prompt, config)
        
        # Clean response (don't truncate judge - we need full verdict)
        response = clean_response(raw_response)
        
        # Validate output format
        is_valid, missing = validate_judge_output(response)
        if not is_valid:
            logger.warning(f"Judge output missing headers: {missing}")
        
        # Extract verdict details from response
        winner = extract_winner_from_text(response) or "tie"
        confidence = extract_confidence_from_text(response) or "medium"
        
        # Create turn record for history
        turn = DebateTurn(
            role="judge",
            phase="verdict",
            round_number=0,
            content=response,
        )
        
        logger.info(f"Judge verdict: {winner} (confidence: {confidence})")
        
        # Return state update with verdict
        return {
            "history": [turn],
            "current_phase": "complete",
            "verdict": {
                "winner": winner,
                "confidence": confidence,
                "reasoning": response,
                "summary": f"The {winner} wins with {confidence} confidence.",
                "proponent_scores": {},
                "opposition_scores": {},
                "key_arguments_proponent": [],
                "key_arguments_opposition": [],
                "ignored_counterarguments": [],
            }
        }
    
    return judge_node


# ============================================================================
# Phase Transition Nodes
# ============================================================================

def create_phase_router(config: DebateConfig):
    """
    Create a router function for phase transitions.
    
    Determines next phase based on current state:
    - opening -> rebuttal (round 1)
    - rebuttal -> rebuttal (if more rounds) or closing
    - closing -> verdict
    - verdict -> END
    """
    def phase_router(state: Dict[str, Any]) -> str:
        """
        Route to next node based on current phase and round.
        
        Returns:
            Name of next node to execute
        """
        phase = state["current_phase"]
        current_round = state["current_round"]
        max_rounds = state.get("max_rounds", config.max_rounds)
        
        # Check last turn to determine speaker
        history = state.get("history", [])
        last_role = history[-1].role if history else None
        
        if phase == "opening":
            if last_role == "opposition":
                # Opening complete, move to rebuttal
                return "start_rebuttal"
            elif last_role == "proponent":
                # Proponent done, opposition's turn
                return "opposition"
            else:
                # First turn
                return "proponent"
        
        elif phase == "rebuttal":
            if last_role == "opposition":
                # Round complete
                if current_round >= max_rounds:
                    return "start_closing"
                else:
                    return "next_round"
            else:
                # Proponent done, opposition's turn
                return "opposition"
        
        elif phase == "closing":
            if last_role == "opposition":
                # Closing complete, go to verdict
                return "judge"
            elif last_role == "proponent":
                # Proponent done, opposition's turn
                return "opposition"
            else:
                return "proponent"
        
        elif phase == "verdict":
            return "end"
        
        # Default
        return "end"
    
    return phase_router


def start_rebuttal_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Transition to rebuttal phase."""
    return {
        "current_phase": "rebuttal",
        "current_round": 1,
    }


def next_round_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Increment round counter."""
    return {
        "current_round": state["current_round"] + 1,
    }


def start_closing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Transition to closing phase."""
    return {
        "current_phase": "closing",
        "current_round": 0,
    }
