"""
Pydantic models for the Multi-Agent Debate System.

Defines typed state structures for LangGraph, ensuring:
- Type safety at runtime
- Immutable history (additive updates only)
- Structured outputs for all agents
"""

from typing import List, Optional, Literal, Annotated
from pydantic import BaseModel, Field
from datetime import datetime
import operator


# ============================================================================
# Agent Response Models
# ============================================================================

class ArgumentSection(BaseModel):
    """A structured section of an argument."""
    heading: str = Field(description="Section heading")
    content: str = Field(description="Section content")


class AgentResponse(BaseModel):
    """
    Structured output from a debate agent.
    
    Enforces consistent format across all agent responses.
    """
    main_argument: str = Field(
        description="Primary argument or thesis statement"
    )
    supporting_points: List[str] = Field(
        default_factory=list,
        description="Supporting evidence or examples"
    )
    rebuttal: Optional[str] = Field(
        default=None,
        description="Response to opponent's previous arguments"
    )
    raw_text: str = Field(
        description="Full formatted response text"
    )


# ============================================================================
# Debate Turn Model
# ============================================================================

class DebateTurn(BaseModel):
    """
    A single turn in the debate.
    
    Captures who spoke, what they said, and when.
    """
    role: Literal["proponent", "opposition", "judge"] = Field(
        description="The agent role that made this turn"
    )
    phase: Literal["opening", "rebuttal", "closing", "verdict"] = Field(
        description="The debate phase of this turn"
    )
    round_number: int = Field(
        ge=0,
        description="Current round number (0 for opening/closing)"
    )
    content: str = Field(
        description="The full response content"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this turn was created"
    )
    word_count: int = Field(
        default=0,
        description="Word count of the response"
    )
    
    def model_post_init(self, __context) -> None:
        """Calculate word count after initialization."""
        if self.word_count == 0:
            self.word_count = len(self.content.split())


# ============================================================================
# Judge Verdict Model
# ============================================================================

class ArgumentScore(BaseModel):
    """Scoring for a single argument dimension."""
    score: int = Field(ge=1, le=10, description="Score from 1-10")
    justification: str = Field(description="Reason for the score")


class JudgeVerdict(BaseModel):
    """
    Structured verdict from the judge agent.
    
    Provides comprehensive analysis with scores and final decision.
    """
    # Overall Verdict
    winner: Literal["proponent", "opposition", "tie"] = Field(
        description="The winning side of the debate"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level in the verdict"
    )
    
    # Detailed Scores
    proponent_scores: dict[str, ArgumentScore] = Field(
        default_factory=dict,
        description="Scores for proponent across dimensions"
    )
    opposition_scores: dict[str, ArgumentScore] = Field(
        default_factory=dict,
        description="Scores for opposition across dimensions"
    )
    
    # Analysis
    key_arguments_proponent: List[str] = Field(
        default_factory=list,
        description="Strongest arguments from proponent"
    )
    key_arguments_opposition: List[str] = Field(
        default_factory=list,
        description="Strongest arguments from opposition"
    )
    ignored_counterarguments: List[str] = Field(
        default_factory=list,
        description="Important points that were not addressed"
    )
    
    # Summary
    reasoning: str = Field(
        description="Detailed explanation of the verdict"
    )
    summary: str = Field(
        description="One-sentence summary of the outcome"
    )


# ============================================================================
# LangGraph State Model
# ============================================================================

def add_turns(existing: List[DebateTurn], new: List[DebateTurn]) -> List[DebateTurn]:
    """
    Reducer function for additive turn updates.
    
    Ensures history is never overwritten, only appended.
    """
    return existing + new


class DebateState(BaseModel):
    """
    The complete state of a debate, used by LangGraph.
    
    Key design decisions:
    - history uses a reducer to ensure additive updates only
    - All fields have defaults for clean initialization
    - Immutable patterns enforced through Pydantic
    """
    # Debate Configuration
    topic: str = Field(
        default="",
        description="The debate topic or proposition"
    )
    max_rounds: int = Field(
        default=3,
        ge=1,
        description="Maximum number of rebuttal rounds"
    )
    
    # State Tracking
    current_round: int = Field(
        default=0,
        ge=0,
        description="Current rebuttal round (0 = opening/closing)"
    )
    current_phase: Literal["opening", "rebuttal", "closing", "verdict", "complete"] = Field(
        default="opening",
        description="Current phase of the debate"
    )
    
    # Debate History (additive only via reducer)
    # Note: In LangGraph, we use Annotated with operator.add for reducers
    # Here we define the type; the reducer is applied in graph.py
    history: List[DebateTurn] = Field(
        default_factory=list,
        description="Complete history of debate turns"
    )
    
    # Final Output
    verdict: Optional[JudgeVerdict] = Field(
        default=None,
        description="Judge's final verdict (set at end)"
    )
    
    # Error Tracking
    errors: List[str] = Field(
        default_factory=list,
        description="Any errors encountered during debate"
    )
    
    def get_last_turn(self, role: Optional[str] = None) -> Optional[DebateTurn]:
        """Get the most recent turn, optionally filtered by role."""
        if not self.history:
            return None
        if role:
            for turn in reversed(self.history):
                if turn.role == role:
                    return turn
            return None
        return self.history[-1]
    
    def get_turns_by_role(self, role: str) -> List[DebateTurn]:
        """Get all turns for a specific role."""
        return [t for t in self.history if t.role == role]
    
    def get_turns_by_phase(self, phase: str) -> List[DebateTurn]:
        """Get all turns for a specific phase."""
        return [t for t in self.history if t.phase == phase]
    
    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
