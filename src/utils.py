"""
Utility functions for the Multi-Agent Debate System.

Provides production safeguards:
- Output validation and format enforcement
- Response length truncation
- Retry logic with exponential backoff
- History formatting for context windows
"""

import re
import time
import logging
from typing import Callable, TypeVar, Any, Optional
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================================
# Retry Logic
# ============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay cap (seconds)
        exponential_base: Base for exponential backoff calculation
        retryable_exceptions: Tuple of exception types to retry
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise
                    
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
            
            raise last_exception  # Should never reach here
            
        return wrapper
    return decorator


# ============================================================================
# Output Validation
# ============================================================================

def validate_structured_output(
    content: str,
    required_headers: list[str],
    strict: bool = False
) -> tuple[bool, list[str]]:
    """
    Validate that response contains required structured headers.
    
    Args:
        content: The response content to validate
        required_headers: List of header strings that must be present
        strict: If True, raise exception on validation failure
        
    Returns:
        Tuple of (is_valid, list of missing headers)
    """
    missing = []
    content_lower = content.lower()
    
    for header in required_headers:
        # Check for markdown header format (## Header or # Header)
        header_patterns = [
            f"## {header.lower()}",
            f"# {header.lower()}",
            f"**{header.lower()}**",
        ]
        if not any(pattern in content_lower for pattern in header_patterns):
            missing.append(header)
    
    is_valid = len(missing) == 0
    
    if strict and not is_valid:
        raise ValueError(f"Response missing required headers: {missing}")
    
    return is_valid, missing


def validate_proponent_output(content: str) -> tuple[bool, list[str]]:
    """Validate proponent response format."""
    required = ["Main Argument", "Supporting Evidence", "Key Takeaway"]
    return validate_structured_output(content, required)


def validate_opposition_output(content: str) -> tuple[bool, list[str]]:
    """Validate opposition response format."""
    required = ["Counter-Argument", "Critical Analysis", "Key Takeaway"]
    return validate_structured_output(content, required)


def validate_judge_output(content: str) -> tuple[bool, list[str]]:
    """Validate judge response format."""
    required = ["Argument Analysis", "Scores", "Verdict", "Reasoning"]
    return validate_structured_output(content, required)


# ============================================================================
# Response Processing
# ============================================================================

def truncate_response(content: str, max_words: int) -> str:
    """
    Truncate response to maximum word count while preserving structure.
    
    Args:
        content: Response content to truncate
        max_words: Maximum allowed word count
        
    Returns:
        Truncated content (may be unchanged if under limit)
    """
    words = content.split()
    
    if len(words) <= max_words:
        return content
    
    # Truncate and add indicator
    truncated_words = words[:max_words]
    truncated_content = " ".join(truncated_words)
    
    # Try to end at a sentence boundary
    last_period = truncated_content.rfind(".")
    if last_period > len(truncated_content) * 0.7:  # At least 70% of content
        truncated_content = truncated_content[:last_period + 1]
    
    logger.info(f"Response truncated from {len(words)} to {len(truncated_content.split())} words")
    
    return truncated_content


def clean_response(content: str) -> str:
    """
    Clean up response content for consistency.
    
    - Remove excessive whitespace
    - Normalize line breaks
    - Strip leading/trailing whitespace
    """
    # Normalize line breaks
    content = content.replace("\r\n", "\n")
    
    # Remove excessive blank lines (more than 2)
    content = re.sub(r"\n{3,}", "\n\n", content)
    
    # Strip and return
    return content.strip()


# ============================================================================
# History Formatting
# ============================================================================

def format_history_for_context(
    history: list,
    max_turns: int = 6,
    max_chars_per_turn: int = 500
) -> str:
    """
    Format debate history for inclusion in prompts.
    
    Args:
        history: List of DebateTurn objects
        max_turns: Maximum recent turns to include
        max_chars_per_turn: Character limit per turn summary
        
    Returns:
        Formatted history string
    """
    if not history:
        return "[No previous turns]"
    
    # Take most recent turns
    recent = history[-max_turns:]
    
    formatted_parts = []
    for turn in recent:
        content_preview = turn.content[:max_chars_per_turn]
        if len(turn.content) > max_chars_per_turn:
            content_preview += "..."
        
        formatted_parts.append(
            f"**{turn.role.upper()}** ({turn.phase}, Round {turn.round_number}):\n"
            f"{content_preview}\n"
        )
    
    return "\n---\n".join(formatted_parts)


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


# ============================================================================
# Output Formatting
# ============================================================================

def format_debate_output(state: Any) -> str:
    """
    Format the complete debate state for display.
    
    Args:
        state: DebateState object
        
    Returns:
        Formatted string for display
    """
    parts = [
        "=" * 60,
        f"DEBATE: {state.topic}",
        "=" * 60,
        ""
    ]
    
    for turn in state.history:
        parts.append(f"\n{'─' * 40}")
        parts.append(f"📢 {turn.role.upper()} - {turn.phase.upper()}", )
        if turn.round_number > 0:
            parts.append(f"   Round {turn.round_number}")
        parts.append(f"   ({turn.word_count} words)")
        parts.append("─" * 40)
        parts.append(turn.content)
        parts.append("")
    
    if state.verdict:
        parts.append("\n" + "=" * 60)
        parts.append("⚖️  FINAL VERDICT")
        parts.append("=" * 60)
        parts.append(f"\nWinner: {state.verdict.winner.upper()}")
        parts.append(f"Confidence: {state.verdict.confidence}")
        parts.append(f"\n{state.verdict.reasoning}")
        parts.append(f"\n📝 {state.verdict.summary}")
    
    return "\n".join(parts)


def extract_winner_from_text(content: str) -> Optional[str]:
    """
    Extract winner from judge's response text.
    
    Args:
        content: Judge's response text
        
    Returns:
        'proponent', 'opposition', 'tie', or None if not found
    """
    content_lower = content.lower()
    
    # Look for explicit winner declaration
    patterns = [
        r"\*\*winner:\s*(proponent|opposition|tie)\*\*",
        r"winner:\s*(proponent|opposition|tie)",
        r"the\s+winner\s+is\s+(?:the\s+)?(proponent|opposition|tie)",
        r"(?:i\s+)?declare\s+(?:the\s+)?(proponent|opposition|tie)\s+(?:as\s+)?(?:the\s+)?winner",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content_lower)
        if match:
            return match.group(1)
    
    return None


def extract_confidence_from_text(content: str) -> Optional[str]:
    """
    Extract confidence level from judge's response text.
    
    Args:
        content: Judge's response text
        
    Returns:
        'high', 'medium', 'low', or None if not found
    """
    content_lower = content.lower()
    
    patterns = [
        r"\*\*confidence:\s*(high|medium|low)\*\*",
        r"confidence:\s*(high|medium|low)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content_lower)
        if match:
            return match.group(1)
    
    return "medium"  # Default if not found
