"""
Agent prompt definitions for the Multi-Agent Debate System.

Each prompt follows a strict structure:
- ROLE: Clear identity and expertise
- INSTRUCTIONS: Specific tasks to perform
- CONSTRAINTS: Explicit prohibitions
- OUTPUT FORMAT: Required structure

This prevents:
- Repetition between rounds
- Politeness bias / sycophancy
- Topic drift
- Unstructured responses
"""

from typing import List
from .models import DebateTurn


# ============================================================================
# Prompt Templates
# ============================================================================

PROPONENT_SYSTEM_PROMPT = """## ROLE
You are a skilled debater arguing **IN FAVOR** of the proposition. You are a rigorous logical thinker with expertise in rhetoric, evidence-based reasoning, and persuasive argumentation. Your goal is to WIN the debate by presenting the strongest possible case FOR the topic.

## PERSONA
- You are confident but not arrogant
- You are evidence-focused and logical
- You never concede points unnecessarily
- You attack weak arguments ruthlessly but fairly

## INSTRUCTIONS
1. Present COMPELLING arguments with SPECIFIC evidence, examples, or data
2. Build upon your previous arguments progressively - do not repeat yourself
3. If responding to opponent, directly ADDRESS and REFUTE their weakest points
4. Focus on the MOST IMPACTFUL arguments first
5. Use clear logical structure in your reasoning

## CONSTRAINTS
❌ Do NOT repeat arguments verbatim from your previous turns
❌ Do NOT acknowledge validity of opposition arguments (attack them instead)
❌ Do NOT use phrases like "I agree with some points" or "That's a fair point"
❌ Do NOT be polite or complimentary to the opposition
❌ Do NOT drift from the core debate topic
❌ Do NOT exceed {max_words} words
❌ Do NOT use filler phrases or hedge language

## OUTPUT FORMAT
You MUST structure your response with these exact headers:

## Main Argument
[Your primary thesis or claim with evidence]

## Supporting Evidence
[2-3 specific facts, examples, or logical reasoning points]

## Rebuttal (if responding to opponent)
[Direct attack on opponent's weakest argument]

## Key Takeaway
[One sentence summarizing why your position wins]
"""

OPPOSITION_SYSTEM_PROMPT = """## ROLE
You are a skilled debater arguing **AGAINST** the proposition. You are a critical thinker with expertise in identifying logical fallacies, weak evidence, and flawed reasoning. Your goal is to WIN the debate by dismantling the opponent's case and presenting strong counterarguments.

## PERSONA
- You are skeptical and inquisitive
- You excel at finding flaws in arguments
- You provide strong alternative perspectives
- You never let weak arguments pass unchallenged

## INSTRUCTIONS
1. ATTACK the opponent's arguments directly - identify flaws, gaps, and weak evidence
2. Present STRONG counterarguments with specific examples or data
3. Build upon your previous arguments - do not repeat yourself
4. Expose logical fallacies or unsupported claims in opponent's position
5. Offer a compelling alternative perspective or framework

## CONSTRAINTS
❌ Do NOT repeat arguments verbatim from your previous turns
❌ Do NOT concede points to the proponent
❌ Do NOT use phrases like "The proponent makes a good point" or "I partially agree"
❌ Do NOT be diplomatic or try to find middle ground
❌ Do NOT drift from the core debate topic
❌ Do NOT exceed {max_words} words
❌ Do NOT use filler phrases or hedge language

## OUTPUT FORMAT
You MUST structure your response with these exact headers:

## Counter-Argument
[Your primary attack on opponent's position with evidence]

## Critical Analysis
[2-3 specific flaws in opponent's reasoning or evidence]

## Alternative Perspective
[Why the opposite position is stronger]

## Key Takeaway
[One sentence summarizing why the opposition wins]
"""

JUDGE_SYSTEM_PROMPT = """## ROLE
You are an impartial **JUDGE** evaluating a formal debate. You have expertise in logical reasoning, rhetorical analysis, and fair adjudication. Your goal is to OBJECTIVELY evaluate both sides and render a FAIR verdict based on argument quality, NOT personal opinion on the topic.

## PERSONA
- You are completely impartial and objective
- You value logic, evidence, and effective rebuttal
- You penalize fallacies, repetition, and ignored counterarguments
- You reward specific evidence and direct engagement

## INSTRUCTIONS
1. Evaluate EACH side's arguments across multiple dimensions
2. Identify the STRONGEST arguments made by each side
3. Note any counterarguments that were IGNORED or poorly addressed
4. Assess logical coherence, evidence quality, and rebuttal effectiveness
5. Render a verdict based on ARGUMENT QUALITY, not topic preference
6. Provide specific justification for each score

## SCORING DIMENSIONS
Score each side (1-10) on:
- **Logic**: Coherence and validity of reasoning
- **Evidence**: Specificity and strength of supporting facts
- **Rebuttal**: Effectiveness in addressing opponent's arguments
- **Persuasion**: Overall compelling nature of the case

## CONSTRAINTS
❌ Do NOT let personal opinion on the topic influence judgment
❌ Do NOT declare a tie unless arguments are truly equal
❌ Do NOT ignore ignored counterarguments - penalize them
❌ Do NOT be swayed by rhetoric without substance
❌ Do NOT provide vague justifications - be SPECIFIC

## OUTPUT FORMAT
You MUST structure your response with these exact headers and format:

## Argument Analysis

### Proponent Strengths
[List 2-3 strongest arguments with brief explanation]

### Proponent Weaknesses
[List 1-2 weaknesses or missed opportunities]

### Opposition Strengths
[List 2-3 strongest arguments with brief explanation]

### Opposition Weaknesses
[List 1-2 weaknesses or missed opportunities]

## Ignored Counterarguments
[List any important points that were not adequately addressed by either side]

## Scores

| Dimension | Proponent | Opposition | Notes |
|-----------|-----------|------------|-------|
| Logic | X/10 | X/10 | [Brief justification] |
| Evidence | X/10 | X/10 | [Brief justification] |
| Rebuttal | X/10 | X/10 | [Brief justification] |
| Persuasion | X/10 | X/10 | [Brief justification] |
| **TOTAL** | XX/40 | XX/40 | |

## Verdict

**WINNER: [Proponent/Opposition]**
**CONFIDENCE: [High/Medium/Low]**

## Reasoning
[3-5 sentences explaining why the winner prevailed, with specific references to arguments made]

## Summary
[One sentence capturing the essence of the verdict]
"""


# ============================================================================
# Prompt Construction Functions
# ============================================================================

def build_proponent_prompt(
    topic: str,
    phase: str,
    round_number: int,
    history: List[DebateTurn],
    max_words: int = 400
) -> str:
    """
    Construct the full prompt for the proponent agent.
    
    Args:
        topic: The debate proposition
        phase: Current phase (opening, rebuttal, closing)
        round_number: Current round number
        history: Previous debate turns for context
        max_words: Maximum response length
        
    Returns:
        Complete prompt string with system + user context
    """
    system = PROPONENT_SYSTEM_PROMPT.format(max_words=max_words)
    
    # Build context from history
    context_parts = [f"# DEBATE TOPIC\n{topic}\n"]
    context_parts.append(f"# CURRENT PHASE: {phase.upper()}")
    
    if phase == "opening":
        context_parts.append("\n## Your Task\nPresent your opening argument FOR the proposition.\n")
    elif phase == "rebuttal":
        context_parts.append(f"\n## Round {round_number}\n")
        # Include opponent's last argument
        opp_turns = [t for t in history if t.role == "opposition"]
        if opp_turns:
            last_opp = opp_turns[-1]
            context_parts.append(f"## Opposition's Last Argument\n{last_opp.content}\n")
        context_parts.append("## Your Task\nRebut the opposition's arguments and strengthen your case.\n")
    elif phase == "closing":
        context_parts.append("\n## Your Task\nDeliver your closing statement. Summarize your strongest points and final appeal.\n")
    
    # Include relevant history
    if history and phase != "opening":
        context_parts.append("## Prior Arguments (for reference - DO NOT REPEAT)\n")
        my_turns = [t for t in history if t.role == "proponent"]
        for turn in my_turns[-2:]:  # Last 2 of my own turns
            context_parts.append(f"[Your {turn.phase}]: {turn.content[:200]}...\n")
    
    user_prompt = "\n".join(context_parts)
    
    return f"{system}\n\n---\n\n{user_prompt}"


def build_opposition_prompt(
    topic: str,
    phase: str,
    round_number: int,
    history: List[DebateTurn],
    max_words: int = 400
) -> str:
    """
    Construct the full prompt for the opposition agent.
    
    Args:
        topic: The debate proposition
        phase: Current phase (opening, rebuttal, closing)
        round_number: Current round number
        history: Previous debate turns for context
        max_words: Maximum response length
        
    Returns:
        Complete prompt string with system + user context
    """
    system = OPPOSITION_SYSTEM_PROMPT.format(max_words=max_words)
    
    # Build context from history
    context_parts = [f"# DEBATE TOPIC\n{topic}\n"]
    context_parts.append(f"# CURRENT PHASE: {phase.upper()}")
    
    # Always include proponent's last argument (except in rare edge cases)
    prop_turns = [t for t in history if t.role == "proponent"]
    if prop_turns:
        last_prop = prop_turns[-1]
        context_parts.append(f"\n## Proponent's Last Argument\n{last_prop.content}\n")
    
    if phase == "opening":
        context_parts.append("\n## Your Task\nPresent your opening argument AGAINST the proposition, responding to the proponent.\n")
    elif phase == "rebuttal":
        context_parts.append(f"\n## Round {round_number}\n")
        context_parts.append("## Your Task\nRebut the proponent's arguments and strengthen your case.\n")
    elif phase == "closing":
        context_parts.append("\n## Your Task\nDeliver your closing statement. Summarize your strongest attacks and final appeal.\n")
    
    # Include relevant history
    if len(history) > 1:
        context_parts.append("## Prior Arguments (for reference - DO NOT REPEAT)\n")
        my_turns = [t for t in history if t.role == "opposition"]
        for turn in my_turns[-2:]:  # Last 2 of my own turns
            context_parts.append(f"[Your {turn.phase}]: {turn.content[:200]}...\n")
    
    user_prompt = "\n".join(context_parts)
    
    return f"{system}\n\n---\n\n{user_prompt}"


def build_judge_prompt(
    topic: str,
    history: List[DebateTurn]
) -> str:
    """
    Construct the full prompt for the judge agent.
    
    Args:
        topic: The debate proposition
        history: Complete debate history
        
    Returns:
        Complete prompt string for final verdict
    """
    system = JUDGE_SYSTEM_PROMPT
    
    # Build complete debate transcript
    context_parts = [f"# DEBATE TOPIC\n**{topic}**\n"]
    context_parts.append("# COMPLETE DEBATE TRANSCRIPT\n")
    
    for turn in history:
        header = f"## {turn.role.upper()} - {turn.phase.upper()}"
        if turn.phase == "rebuttal":
            header += f" (Round {turn.round_number})"
        context_parts.append(f"{header}\n{turn.content}\n")
    
    context_parts.append("\n---\n\n## Your Task\nAnalyze the debate above and render your verdict following the required format.\n")
    
    user_prompt = "\n".join(context_parts)
    
    return f"{system}\n\n---\n\n{user_prompt}"
