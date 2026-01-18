#!/usr/bin/env python3
"""
Multi-Agent Debate System - Main Entry Point

A production-grade multi-agent debate system using LangChain and LangGraph
where multiple LLM agents debate a topic and produce a structured verdict.

Usage:
    python main.py --topic "Your debate topic here" --rounds 3
    
    OR with streaming:
    python main.py --topic "Your topic" --rounds 2 --stream

Requirements:
    - Set OPENROUTER_API_KEY in .env file
    - Install dependencies: pip install -r requirements.txt
"""

import argparse
import logging
import sys
from typing import Optional

from src.config import DebateConfig, get_default_config
from src.graph import run_debate, stream_debate
from src.utils import format_debate_output
from src.models import DebateTurn


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Reduce noise from HTTP libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


# ============================================================================
# Output Formatting
# ============================================================================

def print_turn(turn: DebateTurn) -> None:
    """Print a single debate turn with formatting."""
    # Role emoji mapping
    emojis = {
        "proponent": "🟢",
        "opposition": "🔴",
        "judge": "⚖️",
    }
    
    emoji = emojis.get(turn.role, "📝")
    phase_display = turn.phase.upper()
    
    if turn.round_number > 0:
        phase_display += f" (Round {turn.round_number})"
    
    print(f"\n{'─' * 60}")
    print(f"{emoji} {turn.role.upper()} - {phase_display}")
    print(f"   Words: {turn.word_count}")
    print(f"{'─' * 60}")
    print(turn.content)


def print_verdict(state: dict) -> None:
    """Print the final verdict in a formatted way."""
    verdict = state.get("verdict")
    if not verdict:
        print("\n⚠️  No verdict was produced.")
        return
    
    print("\n" + "=" * 60)
    print("⚖️  FINAL VERDICT")
    print("=" * 60)
    
    # Winner
    winner_emoji = "🟢" if verdict["winner"] == "proponent" else "🔴" if verdict["winner"] == "opposition" else "⚪"
    print(f"\n{winner_emoji} WINNER: {verdict['winner'].upper()}")
    print(f"📊 CONFIDENCE: {verdict['confidence'].upper()}")
    
    print(f"\n📝 SUMMARY: {verdict['summary']}")


def print_debate_summary(state: dict) -> None:
    """Print a summary of the debate."""
    history = state.get("history", [])
    
    print("\n" + "=" * 60)
    print("📊 DEBATE SUMMARY")
    print("=" * 60)
    
    print(f"\n📋 Topic: {state['topic']}")
    print(f"🔄 Rounds: {state['max_rounds']}")
    print(f"📝 Total Turns: {len(history)}")
    
    # Word counts by role
    proponent_words = sum(t.word_count for t in history if t.role == "proponent")
    opposition_words = sum(t.word_count for t in history if t.role == "opposition")
    
    print(f"\n🟢 Proponent Words: {proponent_words}")
    print(f"🔴 Opposition Words: {opposition_words}")


# ============================================================================
# Main Execution
# ============================================================================

def run_standard(topic: str, rounds: int, config: DebateConfig) -> dict:
    """
    Run debate in standard (non-streaming) mode.
    
    Returns complete state after all turns are finished.
    """
    print(f"\n🎯 Starting debate: '{topic}'")
    print(f"   Max rounds: {rounds}")
    print(f"   Model: {config.model_name}")
    print("\n⏳ Running debate (this may take a few minutes)...\n")
    
    final_state = run_debate(topic=topic, max_rounds=rounds, config=config)
    
    # Print all turns
    for turn in final_state["history"]:
        print_turn(turn)
    
    # Print verdict
    print_verdict(final_state)
    
    # Print summary
    print_debate_summary(final_state)
    
    return final_state


def run_streaming(topic: str, rounds: int, config: DebateConfig) -> dict:
    """
    Run debate in streaming mode.
    
    Prints each turn as it completes.
    """
    print(f"\n🎯 Starting streaming debate: '{topic}'")
    print(f"   Max rounds: {rounds}")
    print(f"   Model: {config.model_name}")
    print("\n⏳ Debate in progress...\n")
    
    final_state = None
    
    for event in stream_debate(topic=topic, max_rounds=rounds, config=config):
        # Each event is a dict with node name as key
        for node_name, node_output in event.items():
            # Print new turns as they appear
            new_turns = node_output.get("history", [])
            for turn in new_turns:
                print_turn(turn)
            
            # Capture final state from judge
            if node_name == "judge":
                final_state = node_output
    
    # Reconstruct full state for summary
    if final_state and "verdict" in final_state:
        print_verdict({"verdict": final_state["verdict"]})
    
    return final_state or {}


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Debate System using LangChain and LangGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --topic "AI will replace most jobs in 10 years"
  python main.py --topic "Remote work is better than office work" --rounds 2
  python main.py --topic "Social media is harmful" --rounds 3 --stream --verbose
        """,
    )
    
    parser.add_argument(
        "--topic", "-t",
        type=str,
        required=True,
        help="The debate topic or proposition",
    )
    
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=2,
        help="Number of rebuttal rounds (default: 2)",
    )
    
    parser.add_argument(
        "--stream", "-s",
        action="store_true",
        help="Enable streaming mode (print turns as they complete)",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Override the default model (e.g., 'anthropic/claude-3.5-sonnet')",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Load configuration
    try:
        config = get_default_config()
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\n💡 Tip: Copy .env.example to .env and set your OPENROUTER_API_KEY")
        return 1
    
    # Override model if specified
    if args.model:
        config = DebateConfig(**{**config.model_dump(), "model_name": args.model})
    
    # Run debate
    try:
        if args.stream:
            run_streaming(args.topic, args.rounds, config)
        else:
            run_standard(args.topic, args.rounds, config)
        
        print("\n✅ Debate completed successfully!\n")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Debate interrupted by user.")
        return 130
    
    except Exception as e:
        logging.exception("Debate failed with error")
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
