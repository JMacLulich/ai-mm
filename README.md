# Claude Multi-Model (MM) Tool

Multi-model AI code review and planning CLI tool with parallel execution support.

## Features

- **Parallel Multi-Model Reviews**: Run GPT + Gemini simultaneously for 2x faster feedback
- **Smart Caching**: 24hr TTL response cache with atomic writes
- **Cost Tracking**: Every API call logged with detailed analytics
- **Auto-Retry**: Exponential backoff for transient failures
- **Multiple Models**: GPT-5.2, GPT-5.2-instant, Gemini, Claude

## Installation

```bash
git clone https://github.com/JMacLulich/claude-mm-tool
cd claude-mm-tool
./run install
```

## Usage

```bash
# Multi-model parallel review (fastest)
git diff | ai review --model mm

# Single model review
git diff | ai review --model gpt-5.2-instant --focus security

# Planning
ai plan "Add user authentication"

# Multi-round stabilized planning
ai stabilize "Design rate limiting" --rounds 2

# Cost tracking
ai usage --week

# Cache management
ai cache stats
ai cache clear --older-than 24
```

## Development

```bash
# Lint code
./run lint
./run lint fix

# Run tests
./run test
./run test unit
./run test integration

# Install locally
./run install
```

## Architecture

```
claude-mm-tool/
├── src/claude_mm/          # Python package
│   ├── api.py              # API module interface
│   ├── cache.py            # Response caching (atomic writes)
│   ├── costs.py            # Cost estimation & pricing
│   ├── config.py           # Configuration management
│   ├── retry.py            # Exponential backoff
│   ├── usage.py            # Cost logging & analytics
│   └── providers/          # Model provider implementations
├── bin/
│   └── ai                  # Main CLI (modular architecture)
├── tests/
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── commands/               # ./run command modules
│   ├── lint/
│   ├── test/
│   ├── install/
│   └── help/
└── .carapace/              # Shell completions
```

## Configuration

Create `~/.config/claude-mm-tool/env`:

```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_AI_API_KEY="..."
export ANTHROPIC_API_KEY="sk-ant-..."  # optional
```

## Testing

```bash
# Unit tests (no API keys needed)
./run test unit

# Integration tests (requires API keys)
export OPENAI_API_KEY="sk-..."
export GOOGLE_AI_API_KEY="..."
./run test integration

# All tests with coverage
./run test
```

## Design Principles

- **Single Responsibility**: Each module has one clear purpose
- **Thread-Safe**: Atomic writes, file locking
- **Observable**: All API calls logged with cost
- **Fail-Safe**: Auto-retry with smart error detection
- **Fast**: Parallel execution, response caching

## License

MIT
