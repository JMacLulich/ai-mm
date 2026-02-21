# Claude Multi-Model (MM) Tool

Multi-model AI code review and planning CLI tool with parallel execution support.

## Features

- **Parallel Multi-Model Reviews**: Run GPT + Gemini + Claude Opus simultaneously for broader feedback
- **Smart Caching**: 24hr TTL response cache with atomic writes
- **Cost Tracking**: Every API call logged with detailed analytics
- **Auto-Retry**: Exponential backoff for transient failures
- **Multiple Models**: GPT-5.2, GPT-5.2-instant, Gemini, Claude

## Installation

### Quick Start

```bash
git clone https://github.com/JMacLulich/ai-mm
cd ai-mm
./run install
```

### What Happens During Installation

The installer will:

1. **Create a Python virtual environment** at `~/.local/venvs/ai/`
2. **Install dependencies** (openai, google-generativeai, anthropic, pytest, ruff)
3. **Install the `ai` command** to `~/.local/bin/ai`
4. **Install shell completions** (if Carapace is installed)
5. **Prompt for API keys** interactively

### API Keys Configuration

During installation, you'll be prompted for:

- **OpenAI API key** (required for GPT models) - Get from https://platform.openai.com/api-keys
- **Google AI API key** (required for Gemini models) - Get from https://aistudio.google.com/apikey
- **Anthropic API key** (optional, for Claude models) - Get from https://console.anthropic.com/

**You can press Enter to skip any key** and add them later.

API keys are stored securely at `~/.config/ai-mm/env` with restricted permissions (chmod 600).

### Adding or Updating API Keys Later

If you skipped API keys during installation or need to add missing ones:

```bash
./run install --keys
```

**What this does:**
- Checks which keys are already configured
- Shows ✓ for keys you have
- Only prompts for missing keys
- Never overwrites existing keys

**Example:**
```bash
$ ./run install --keys
⚙️  Checking for missing API keys...

✓ Google AI API key already configured
Enter your OpenAI API key (or press Enter to skip): sk-proj-...
✓ Anthropic API key already configured

✓ API keys saved to ~/.config/ai-mm/env
```

## Usage

```bash
# Multi-model parallel review (fastest)
git diff | ai review --model mm

# Single model review
git diff | ai review --model gpt-5.2-chat-latest --focus security

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
ai-mm/
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

### API Keys Location

API keys are stored at: `~/.config/ai-mm/env`

**Recommended:** Use the installer to configure keys interactively:

```bash
./run install --keys
```

**Manual setup:** Create the file manually:

```bash
# Create the config directory
mkdir -p ~/.config/ai-mm

# Create the env file
cat > ~/.config/ai-mm/env <<'EOF'
export OPENAI_API_KEY="sk-..."
export GOOGLE_AI_API_KEY="..."
export ANTHROPIC_API_KEY="sk-ant-..."  # optional
EOF

# Set secure permissions
chmod 600 ~/.config/ai-mm/env
```

**Note:** The installer automatically sets proper permissions (chmod 600) to keep your keys secure.

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

## Acknowledgements

The `./run` system was inspired by https://github.com/alesya-h/ - thank you Alesya!

## License

MIT
