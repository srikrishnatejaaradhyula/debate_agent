# Multi-Agent Debate System

A production-grade multi-agent debate system using **LangChain**, **LangGraph**, and **OpenRouter** where multiple LLM agents debate a topic and produce a structured verdict.

## Features

- **Multi-Agent Architecture**: Proponent, Opposition, and Judge agents with distinct personas
- **Structured Prompts**: Role/Instructions/Constraints/Output format prevents repetition and drift
- **State Management**: Typed LangGraph state with additive history (immutable)
- **Production Safeguards**: Retry logic, output validation, length caps
- **Flexible Execution**: Both synchronous and streaming modes
- **Extensible Design**: Easy to add new agents or debate phases

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenRouter API key
# OPENROUTER_API_KEY=your_key_here
```

### 3. Run a Debate

```bash
# Basic usage
python main.py --topic "AI will replace most jobs in 10 years"

# With custom rounds
python main.py --topic "Remote work is better than office work" --rounds 3

# Streaming mode (see turns as they complete)
python main.py --topic "Social media is harmful" --stream

# Verbose logging
python main.py --topic "Climate change is reversible" --verbose
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Proponent  │────▶│  Opposition │────▶│   Judge    │
│   Agent     │     │    Agent    │     │   Agent     │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌──────────────────────────────────────────────────────┐
│                    LangGraph State                   │
│  • topic          • current_phase    • history (add) │
│  • max_rounds     • current_round    • verdict       │
└──────────────────────────────────────────────────────┘
```

### Debate Flow

1. **Opening Round**
   - Proponent presents opening argument FOR the topic
   - Opposition responds with opening argument AGAINST

2. **Rebuttal Rounds** (configurable, default: 2)
   - Proponent rebuts opposition's points
   - Opposition rebuts proponent's points
   - Repeat for N rounds

3. **Closing Round**
   - Proponent delivers closing statement
   - Opposition delivers closing statement

4. **Verdict**
   - Judge analyzes all arguments
   - Produces structured verdict with scores

## Project Structure

```
debate_agent/
├── src/
│   ├── __init__.py      # Package exports
│   ├── config.py        # Configuration management
│   ├── models.py        # Pydantic state models
│   ├── prompts.py       # Agent prompt templates
│   ├── agents.py        # Agent node implementations
│   ├── graph.py         # LangGraph orchestration
│   └── utils.py         # Utilities and safeguards
├── main.py              # CLI entry point
├── requirements.txt     # Dependencies
├── .env.example         # Environment template
└── README.md            # This file
```

## Agent Prompt Design

Each agent follows a strict prompt structure:

```
## ROLE
[Who the agent is and their expertise]

## INSTRUCTIONS
[Specific tasks to perform, numbered]

## CONSTRAINTS
[Explicit prohibitions with ❌ markers]

## OUTPUT FORMAT
[Required headers for structured output]
```

### Key Constraints

- **No Repetition**: Agents cannot repeat previous arguments
- **No Sycophancy**: Agents cannot praise opponent's points
- **No Topic Drift**: Must stay focused on the debate topic
- **Length Limits**: Responses capped at configurable word count

## Extending the System

### Add a New Agent

1. **Create prompt** in `src/prompts.py`:
   ```python
   FACT_CHECKER_SYSTEM_PROMPT = """## ROLE..."""
   
   def build_fact_checker_prompt(...):
       ...
   ```

2. **Create node** in `src/agents.py`:
   ```python
   def create_fact_checker_node(config):
       def fact_checker_node(state):
           ...
       return fact_checker_node
   ```

3. **Wire into graph** in `src/graph.py`:
   ```python
   graph.add_node("fact_checker", create_fact_checker_node(config))
   graph.add_edge("fact_checker", "next_node")
   ```

### Modify Debate Flow

Edit `src/graph.py` to:
- Add new phases
- Change routing logic
- Insert nodes between existing edges

### Change LLM Model

Set in `.env`:
```
DEFAULT_MODEL=openai/gpt-4-turbo
```

Or via CLI:
```bash
python main.py --topic "Topic" --model "anthropic/claude-3-opus"
```

## API Usage

```python
from src.graph import run_debate, stream_debate
from src.config import DebateConfig

# Run synchronously
final_state = run_debate(
    topic="AI will replace most jobs",
    max_rounds=2
)

# Access results
for turn in final_state["history"]:
    print(f"{turn.role}: {turn.content}")

print(f"Winner: {final_state['verdict']['winner']}")

# Stream execution
for event in stream_debate(topic="Remote work is better", max_rounds=2):
    print(event)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OPENROUTER_API_KEY` | (required) | Your OpenRouter API key |
| `DEFAULT_MODEL` | `anthropic/claude-3.5-sonnet` | LLM model to use |
| `MAX_ROUNDS` | `3` | Maximum rebuttal rounds |
| `MAX_RESPONSE_LENGTH` | `500` | Max words per response |

## Troubleshooting

### "API key not configured"
- Ensure `.env` file exists with valid `OPENROUTER_API_KEY`
- Check the key is not the placeholder text

### "Max retries exceeded"
- Check your OpenRouter account has credits
- Verify the model name is correct
- Try a different model

### Output not structured
- This is logged as a warning but doesn't stop execution
- Check the prompt templates in `src/prompts.py`

## License

MIT License - See LICENSE file for details.
