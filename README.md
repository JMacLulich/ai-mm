# AI MM - Multi-Model Code Review Tool

Get code reviews from GPT, Gemini, Claude, and local LLMs (Ollama) - in parallel or individually.

## Why AI MM?

- **Broader feedback**: Different models catch different issues. Run them in parallel, get consolidated reviews.
- **Works offline**: Use Ollama with Qwen or Llama for free, private reviews on your machine.
- **Cost-aware**: Every API call tracked, cached responses save money.
- **Architecture-focused**: Reviews check DRY, Single Responsibility, and Least Astonishment principles.

## Installation

```bash
git clone https://github.com/JMacLulich/ai-mm
cd ai-mm
./run install
```

### What Gets Installed

1. Python virtual environment at `~/.local/venvs/ai/`
2. The `ai` command in `~/.local/bin/ai`
3. Shell completions (if Carapace is installed)
4. Interactive API key configuration

### API Keys

Configure during installation or later with:

```bash
ai config  # Interactive TUI for managing keys
```

**Supported providers:**
- **OpenAI** - GPT-5.2 models
- **Google** - Gemini 3 Pro
- **Anthropic** - Claude Opus 4.5
- **Ollama** - Local LLMs (no key needed)

Keys stored at `~/.config/ai-mm/env` with secure permissions.

## Usage

```bash
# Parallel multi-model review (GPT + Gemini + Claude)
git diff | ai review --model mm

# Fast models only (cheaper)
git diff | ai review --model fast

# Single model
git diff | ai review --model gpt --focus security
git diff | ai review --model gemini --focus performance

# Local LLM (free, offline)
git diff | ai review --model ollama

# Architecture review
git diff | ai review --model mm --focus architecture

# Planning
ai plan "Add user authentication"

# Multi-round stabilized planning
ai stabilize "Design rate limiting" --rounds 2

# Check costs
ai usage --week

# Manage cache
ai cache stats
ai cache clear
```

## Local LLM Support

Use Ollama for free, private code reviews:

```bash
# Install Ollama
brew install ollama
ollama pull qwen2.5:14b

# Review with local model
git diff | ai review --model ollama
```

No API key needed. Works offline. Your code never leaves your machine.

## Development

```bash
./run lint        # Check code quality
./run lint fix    # Auto-fix issues
./run test        # Run all tests
./run test unit   # Unit tests only
./run install     # Reinstall after changes
```

## Architecture

```
ai-mm/
├── src/claude_mm/
│   ├── api.py              # Review and plan functions
│   ├── cache.py            # Response caching
│   ├── costs.py            # Cost estimation
│   ├── config_tui.py       # Interactive config UI
│   ├── env.py              # API key management
│   ├── prompts.py          # System prompts
│   ├── models.py           # Model registry
│   └── providers/          # OpenAI, Google, Anthropic, Ollama
├── bin/ai                  # CLI entry point
├── tests/                  # Unit and integration tests
└── commands/               # ./run commands
```

## Configuration

```bash
# Interactive config UI
ai config

# Manual setup
mkdir -p ~/.config/ai-mm
cat > ~/.config/ai-mm/env <<'EOF'
export OPENAI_API_KEY="sk-..."
export GOOGLE_AI_API_KEY="..."
export ANTHROPIC_API_KEY="sk-ant-..."
# No key needed for Ollama
EOF
chmod 600 ~/.config/ai-mm/env
```

## Design Principles

- **Single Responsibility**: Each module does one thing well
- **Thread-Safe**: Atomic writes, file locking for parallel operations
- **Observable**: All API calls logged with costs
- **Fail-Safe**: Auto-retry with exponential backoff
- **Fast**: Parallel execution, response caching

## License

MIT
