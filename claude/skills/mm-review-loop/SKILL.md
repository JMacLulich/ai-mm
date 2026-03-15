---
name: mm-review-loop
description: Run iterative multi-model code reviews, fixing issues each round, until only low-priority (nice-to-have) items remain. Use when user says "do multiple mm rounds till low hanging left", "mm review loop", "keep reviewing and fixing", "iterate till clean", or "review and fix until only nits left".
disable-model-invocation: false
allowed-tools: Bash(git *), Bash(which ai), Bash(ai review *), Bash(./run *), Read, Edit, Write, Glob, Grep
---

# MM Review Loop Skill

## Purpose

Run mm review → fix all must-fix/high/medium issues → repeat until models only flag low-priority (nice-to-have) items. This is the iterative improvement workflow.

## Trigger Phrases

- "do multiple mm rounds till low hanging left"
- "mm review loop"
- "keep reviewing and fixing"
- "iterate till clean"
- "review and fix until only nits left"
- "run mm rounds until done"
- "loop mm review"

## Priority Classification

When reading review output, classify each flagged issue:

| Level | When to fix | Examples |
|-------|-------------|---------|
| 🔴 Must-fix | Always fix | Crash bugs, security holes, API contract violations, data loss |
| 🟠 High | Fix in this session | Performance degradation, correctness bugs, missing validation |
| 🟡 Medium | Fix if quick | Code smell, minor inconsistency, redundant code |
| 🔵 Low / Nice-to-have | Stop here | Tests, architectural refactoring, style preferences, "consider..." suggestions |

**Stop condition:** All models' "must-fix" / "top issues" are 🔵 Low or Nice-to-have only.

Items that are ALWAYS low-priority (accept and move on):
- "Add tests" / "no tests included"
- "Extract shared logic / reduce duplication"
- Fundamental platform limitations (e.g., "Python threads can't be cancelled")
- "Consider a typed dataclass instead of dict"
- "Make X configurable"
- Hardcoded model names / magic strings
- Code duplication between sync/async paths

## Workflow

### Step 1: Determine Target Files

Ask user: "What files should I review?" or infer from recent changes via:
```bash
git diff --stat
```

Build the review command from the target files, e.g.:
```bash
cat src/foo/api.py src/foo/cache.py | ai review --model mm --focus review --per-model-timeout 120
```

### Step 2: Run MM Review

Execute the review and capture output. Read the full output file if it's saved to disk.

### Step 3: Triage All Findings

For each finding across ALL models:
1. Classify as 🔴/🟠/🟡/🔵
2. Check for consensus (multiple models flagging = higher priority)
3. Note if it's already been fixed in a previous round

Build a priority table:
```
| Priority | Issue | Models | Action |
|----------|-------|--------|--------|
| 🔴 | [issue] | GPT + Gemini | Fix now |
| 🟠 | [issue] | Opus | Fix now |
| 🟡 | [issue] | Gemini | Fix if quick |
| 🔵 | No tests | All | Skip |
```

### Step 4: Fix All Non-Low-Priority Issues

Fix every 🔴, 🟠, and 🟡 issue:
1. Read the relevant files
2. Make targeted edits
3. Run `./run lint fix && ./run lint` after each batch
4. Run `./run test` to verify nothing broke
5. Commit the fixes: `git commit -m "fix: address round N findings"`

### Step 5: Check Stop Condition

After fixing, evaluate:
- **If ALL remaining findings are 🔵 Low** → STOP, declare done
- **If any 🔴/🟠/🟡 remain** → Go to Step 2, run another review round

### Step 6: Final Report

When done, present:
```
## MM Review Loop Complete

**Rounds run:** N
**Issues fixed:** M

### Remaining (Low Priority / Nice-to-have)
- [list of accepted low-priority items]

### Not fixed (by design)
- Thread cancellation: fundamental Python limitation (documented)
- Tests: separate task
- [other architectural items]
```

## Key Principles

**Be decisive about priority.** Don't chase architectural preferences forever. The goal is a well-functioning codebase, not a perfect one.

**Watch for model hallucinations.** Models sometimes flag issues that don't exist or re-flag already-fixed items. Re-read the code before fixing.

**Track what's truly fixed vs. documented.** Some "must-fix" items can be resolved by adding a docstring or comment explaining the limitation.

**Accept permanent limitations.** Python thread cancellation, sync/async duplication, hardcoded constants — these are architectural tradeoffs, not bugs.

## Example Session

```
User: "do multiple mm rounds till low hanging left"

Round 1 → 12 must-fix, 8 high, 5 medium, 3 low
  → Fixed: all 12 must-fix, 8 high, 5 medium
  → Committed: "fix: address round 1 findings"

Round 2 → 4 must-fix, 3 high, 2 medium, 5 low
  → Fixed: all 4 must-fix, 3 high, 2 medium
  → Committed: "fix: address round 2 findings"

Round 3 → 0 must-fix, 1 high, 6 low
  → Fixed: 1 high
  → Committed: "fix: address round 3 findings"

Round 4 → 0 must-fix, 0 high, 8 low
  → ALL items are low priority → STOP

Done! 3 active rounds. Remaining items are all nice-to-haves.
```
