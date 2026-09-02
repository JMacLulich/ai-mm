# AI MM

AI-assisted code review, planning, and stabilization through the host-level Rust
[`llm-router`](../llm-router/README.md) service.

`ai-mm` owns review orchestration and prompts. It does **not** choose API models,
construct vendor clients, hold provider credentials, calculate vendor pricing, or
implement fallback cascades. Every LLM call goes to `llm-router` as a semantic
`stage:` or `profile:` intent.

## Architecture

```text
ai review / ai plan / rig-assist
        |
        | stage:review | stage:audit | profile:local_only | ...
        v
LLMRouterProvider
        |
        | POST http://127.0.0.1:4000/v1/chat/completions
        v
Rust llm-router
        |
        | profile resolution, model/provider selection, retries,
        | fallbacks, request quirks, cost and provenance
        v
provider adapter selected by llm-router
```

Responses without the router provenance block are rejected. This prevents a proxy,
misconfigured endpoint, or direct vendor service from silently bypassing routing policy.

## Prerequisite

Install and run the router from `~/Development/llm-router`:

```bash
command -v llm-router
llm-router --version
curl -fsS http://127.0.0.1:4000/health
```

Provider credentials and exact model policy belong in
`~/.config/llm-router/`, not in the `ai-mm` environment.

## Install

```bash
./run install
```

This installs the `ai` command and its Python environment under `~/.local/`.

## Review

```bash
# Router-owned normal review
git diff | ai review --model stage:review

# Normal single review: fail-closed DeepSeek V4 Flash profile
git diff | ai review --model deepseek --focus verification

# Parallel semantic council seats; each seat is independently routed
git diff | ai review --model mm --focus review

# Explicit opt-in/offline local review; never an automatic fallback
git diff | ai review --model local

# Explicit router profile
git diff | ai review --model profile:kimi --reasoning-effort max
```

`--model` is retained as the CLI option name for compatibility, but its value is a
router selector, not an API model ID. Supported forms are:

- `stage:<intent>` — let the router resolve the stage;
- `profile:<name>` — explicitly request a router-owned profile;
- `deepseek` — stable `stage:audit` selector; `llm-router` currently resolves it
  to fail-closed `profile:deepseek_v4_flash_direct`;
- `local`, `kimi`, and `commercial` — convenience profile selectors;
- `mm`, `all`, `fast`, `local`, and `max` — orchestration groups.

Exact API model names are rejected.

### Review groups

| Group | Router intents |
| --- | --- |
| `mm` | `stage:review`, `stage:audit`, `stage:adversarial` |
| `all` | `mm` plus `stage:architect` |
| `fast` | `stage:review` with a `fast` effort hint |
| `local` | `profile:local_only` |
| `max` | `profile:kimi` with a `max` effort hint |

The group controls parallel review seats only. It never names providers or fallback
models; the router remains authoritative for each seat.

Normal review is one `deepseek` seat after all edits are complete. Use `mm`, `all`,
or an iterative loop only when the operator explicitly requests extra review. Local
review is opt-in/offline tooling and must not be inserted into the critical path.

The historical `lmstudio` alias has been removed intentionally because stale
automation used it to put unbounded local Qwen reviews on the critical path. Use
`local` explicitly when an operator requests an offline review.

### Focus

Use `--focus` to tune the review prompt:

- `review`
- `verification`
- `security`
- `performance`
- `architecture`
- `testing`
- `general`

### Timeouts

`--per-model-timeout` is a client deadline for one complete router task. The default
is 600 seconds so the router has time to execute its bounded cascade. Individual
attempt timeouts remain router-owned.

`--local-model-timeout` is accepted only for backward compatibility and no longer
changes routing or provider timeouts.

## Planning

```bash
ai plan "Refactor billing" --model stage:planning --output-format json
ai plan "Refactor billing" --model mm --depth deep --rounds 3 --strict
```

The planning council uses semantic planning, architecture, and adversarial stages.
Structured response validation and synthesis remain in `ai-mm`; execution routing
remains in Rust.

## Rig assistance

```bash
ai rig-assist --mode plan --model local --input packet.json
ai rig-assist --mode recovery --model deepseek --input packet.json
```

Rig assistance accepts only `local` and `deepseek`. Returned JSON remains
schema-constrained and advisory-only.

## Configuration and health

The only `ai-mm` connection setting is:

```bash
export LLM_ROUTER_BASE_URL="http://127.0.0.1:4000/v1"
```

The default already points there. Use either command to inspect the connection:

```bash
ai config
ai check-models
```

`ai check-models` calls the router health endpoint; it does not send a paid test
completion.

Optional settings:

```yaml
default_models:
  plan: stage:planning
  review: stage:review
review_per_model_timeout_seconds: 600
cache_ttl_hours: 24
```

## Router provenance and cost

`llm-router` returns the resolved profile, provider, served model, escalation state,
and cost source. `ai-mm` exposes that metadata on `ReviewResult.metadata` and records
the router-reported cost. There is no second pricing table or cost estimate in this
repository.

## Development

```bash
./run test
./run lint
```

Architecture tests fail if feature code imports direct vendor providers, contains API
model selection, or expands the config surface beyond the router connection.

## Migration notes

Version 0.8 removes direct OpenAI, Anthropic, DeepSeek, Alibaba, Ollama, and LM Studio
adapters from `ai-mm`. It also removes local provider fallback logic and exact model
groups. Configure those providers and their cascades in `llm-router` instead.
