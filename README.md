# AI MM - Multi-Model Code Review Tool

Get code reviews from GPT, DeepSeek, Claude, Alibaba Cloud Qwen, and local LLMs - in parallel or individually.

## Why AI MM?

- **Broader feedback**: Different models catch different issues. Run them in parallel, get consolidated reviews.
- **Works offline**: Use Ollama with Qwen or Llama for free, private reviews on your machine.
- **Cost-aware**: Every API call tracked, cached responses save money.
- **Architecture-focused**: Reviews check DRY, Single Responsibility, and Least Astonishment principles.
- **Rigorous PR review mode**: `--focus review` enforces staff-level multi-pass risk analysis.
- **Adversarial verification**: DeepSeek V4 Pro can run at max thinking with an evidence-first verification prompt.

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
ai configure  # Alias for interactive TUI
```

**Supported providers:**
- **OpenAI** - GPT-5.6 Sol, GPT-5.4, and GPT-5.2 models
- **DeepSeek** - DeepSeek V4 Pro and Flash (set `DEEPSEEK_API_KEY`)
- **Anthropic** - Claude Opus 4.6
- **Alibaba Cloud DashScope** - Qwen 3.6 cloud (set `DASHSCOPE_API_KEY`)
- **Ollama** - Local LLMs (set `OLLAMA_BASE_URL`)
- **LM Studio** - Local OpenAI-compatible Qwen 3.6 (set `LMSTUDIO_BASE_URL`)

Keys stored at `~/.config/ai-mm/env` with secure permissions.
Review defaults are stored in `~/.config/ai/config.yaml`.
Alibaba Cloud uses DashScope's OpenAI-compatible default endpoint. Override with
`DASHSCOPE_BASE_URL` only if your account requires a custom endpoint.
DeepSeek uses `https://api.deepseek.com` by default. Override with
`DEEPSEEK_BASE_URL` only for a compatible proxy or gateway.

## Usage

```bash
# Parallel multimode review (GPT + DeepSeek + Claude + local Ollama + LM Studio)
git diff | ai review --model mm

# Fast models only (cheaper)
git diff | ai review --model fast

# Single model
git diff | ai review --model gpt --focus security
git diff | ai review --model deepseek --focus performance

# DeepSeek V4 Pro at API-level max thinking with adversarial verification
git diff | ai review --model deepseek-pro-xhigh --focus verification

# Local LLM (free, offline)
git diff | ai review --model ollama

# Qwen 3.6: uses Alibaba Cloud if DASHSCOPE_API_KEY is configured,
# otherwise falls back to local LM Studio
git diff | ai review --model qwen3.6

# GPT-5.6 Sol at xhigh reasoning effort
git diff | ai review --model sol-5.6 --reasoning-effort xhigh --focus review

# Convenience profile for the same Sol xhigh review
git diff | ai review --model sol-xhigh --focus review

# Architecture review
git diff | ai review --model mm --focus architecture

# Rigorous staff-level PR review format
git diff | ai review --model mm --focus review

# Planning
ai plan "Add user authentication"
ai plan "Design resilient background jobs" --depth deep --rounds 3 --strict
ai plan "Refactor billing" --model mm --context auto --output-format json

# Multi-round stabilized planning
ai stabilize "Design rate limiting" --rounds 2

# Check costs
ai usage --week

# Manage cache
ai cache stats
ai cache clear
```

### Review Focus Areas

Use `--focus` to bias what the models prioritize:

- `review` - Rigorous staff-level PR review with structured multi-pass output
- `verification` - Evidence-first adversarial checking with explicit falsification attempts
- `general` - Broad code review across correctness, quality, and risk
- `security` - Vulnerabilities, validation gaps, auth/authz issues
- `performance` - Efficiency, query patterns, allocations, and hot paths
- `architecture` - Design quality, boundaries, coupling, and maintainability
- `testing` - Coverage gaps, edge cases, determinism, and test quality

### GPT-5.6 Sol reasoning effort

`sol-5.6` resolves to the OpenAI API model `gpt-5.6-sol`. Use
`--reasoning-effort xhigh` when you want the deepest supported reasoning pass.
The dedicated `sol-xhigh` profile applies that setting automatically. Sol is
intentionally not part of the default `mm` group because it is a premium model;
invoke it explicitly when the review warrants the additional cost.

### DeepSeek max-thinking verification

`deepseek-pro-xhigh` resolves to `deepseek-v4-pro`, enables thinking, and maps the
shared `xhigh` setting to DeepSeek's `max` reasoning effort. Pair it with
`--focus verification` to require evidence, falsification attempts, and explicit
VERIFIED/DISPROVED/UNVERIFIED verdicts. The standard `mm`, `all`, and `fast`
groups now use DeepSeek as the default independent reasoning provider. The max
profile uses a 300-second default orchestration and HTTP timeout and disables
hidden SDK retries so a timed-out expensive request is not silently submitted again.

### Iterative review loop

For an iterative pass that fixes findings and reruns review until only
low-priority items remain, use the bundled `mm-review-loop` Claude skill or run
the commands directly:

```bash
# Broad multi-model loop
git diff | ai review --model mm --focus review

# Deep adversarial Sol loop
git diff | ai review --model sol-xhigh --focus review --no-cache

# Evidence-first DeepSeek max-thinking loop
git diff | ai review --model deepseek-pro-xhigh --focus verification --no-cache
```

The loop treats critical, high, and valid medium findings as work to fix in the
current round. It stops when remaining findings are low-priority or clearly
non-actionable. Use `--no-cache` when you require a fresh model pass; otherwise
the cache is safe across reasoning levels because the effort setting is part of
the cache key.

## Local LLM Support

Use Ollama or LM Studio for free, private code reviews:

```bash
# Install Ollama
brew install ollama
ollama pull qwen2.5:14b

# Configure Ollama endpoint for ai-mm
ai config
ai configure

# Review with local model
git diff | ai review --model ollama

# Review with local Qwen 3.6 via LM Studio
git diff | ai review --model lmstudio
```

No API key needed, but `OLLAMA_BASE_URL` must be configured. Works offline.
Your code never leaves your machine.

For Qwen 3.6, `ai review --model qwen3.6` selects Alibaba Cloud DashScope when
`DASHSCOPE_API_KEY` is configured. If DashScope is not configured, it uses the
local LM Studio model `qwen/qwen3.6-35b-a3b`.

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
│   └── providers/          # OpenAI, DeepSeek, Anthropic, Alibaba, Ollama, LM Studio
├── bin/ai                  # CLI entry point
├── tests/                  # Unit and integration tests
└── commands/               # ./run commands
```

## Configuration

```bash
# Interactive config UI
ai config
ai configure

# Manual setup
mkdir -p ~/.config/ai-mm
cat > ~/.config/ai-mm/env <<'EOF'
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="..."
export ANTHROPIC_API_KEY="sk-ant-..."
export DASHSCOPE_API_KEY="..."
export OLLAMA_BASE_URL="http://localhost:11434"
export LMSTUDIO_BASE_URL="http://127.0.0.1:1234/v1"
EOF
chmod 600 ~/.config/ai-mm/env

mkdir -p ~/.config/ai
cat > ~/.config/ai/config.yaml <<'EOF'
review_per_model_timeout_seconds: 60
EOF
```

When editing Ollama in `ai config`, the endpoint is shown in plain text (not masked).
If missing, the UI suggests `http://localhost:11434`.
`ai config` also lets you edit the default per-model review timeout.

## Design Principles

- **Single Responsibility**: Each module does one thing well
- **Thread-Safe**: Atomic writes, file locking for parallel operations
- **Observable**: All API calls logged with costs
- **Fail-Safe**: Auto-retry with exponential backoff
- **Fast**: Parallel execution, response caching

## License

MIT
