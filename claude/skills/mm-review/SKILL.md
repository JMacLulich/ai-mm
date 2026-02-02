---
name: mm-review
description: Run parallel multi-model AI code reviews using GPT + Gemini for instant dual perspectives on code changes. Use when user says "mm review", "get a mm review", "multimode review", "run mm review", "review with mm", "get multi-model feedback", or "parallel review".
disable-model-invocation: false
allowed-tools: Bash(git *), Bash(which ai), Bash(ai review *), AskUserQuestion
---

# Multi-Model Review Skill

## Purpose

Run parallel multi-model AI code reviews using the `claude-mm-tool`. Provides instant dual perspectives from GPT + Gemini for faster, broader feedback.

## When to Use

Use this skill when user says:
- "get a mm review"
- "mm review"
- "multimode review"
- "run mm review"
- "review with mm"
- "get multi-model feedback"
- "parallel review"

## Prerequisites

- `claude-mm-tool` must be installed at `~/dev/claude-mm-tool` or `~/../claude-mm-tool`
- API keys configured in `~/.config/claude-mm-tool/env`

## Instructions

### Step 1: Verify Tool Installation

Check if claude-mm-tool is installed:

```bash
which ai
```

If not found, guide user to install:
```bash
cd ~/dev/claude-mm-tool
./run install
```

### Step 2: Detect Changes

Check for uncommitted changes:

```bash
git status --short
```

If changes exist, use `git diff`.
If no changes, ask user what they want to review.

### Step 3: Determine Focus

Ask user what focus area they want:

Use `AskUserQuestion`:
- Question: "What focus do you want for the multi-model review?"
- Header: "Review Focus"
- Options:
  1. "General (Recommended)" - "Comprehensive analysis covering all aspects"
  2. "Security" - "Focus on security vulnerabilities and data safety"
  3. "Performance" - "Focus on performance issues and optimizations"
  4. "Architecture" - "Focus on design patterns and architecture"

### Step 4: Run Multi-Model Review

Execute the mm review:

```bash
git diff | ai review --model mm --focus {chosen_focus}
```

The output will show:
- GPT review results
- Gemini review results
- Displayed side-by-side for comparison

### Step 5: Summarize Results

Present a structured summary:

```markdown
# Multi-Model Review Summary

## GPT Findings
[Summarize GPT's key points - top 3-5 issues]

## Gemini Findings
[Summarize Gemini's key points - top 3-5 issues]

## Consensus Issues
[Issues both models agree on - these are high priority]

## Divergent Perspectives
[Where models disagree - user should investigate further]

## Recommended Actions
1. [Prioritized action items based on consensus]
2. ...
```

### Step 6: Offer Follow-Up

Ask if user wants:
- Detailed explanation of any specific finding
- Help implementing suggested fixes
- Another review with different focus

## Example Interaction

```
User: "get a mm review"

Claude: I'll run a multi-model review. What focus do you want?
[Shows question with 4 options]

User: [Selects "Security"]

Claude: Running parallel security review with GPT + Gemini...
[Executes: git diff | ai review --model mm --focus security]

# Multi-Model Review Summary

## GPT Findings
1. SQL injection risk in user_id parameter (line 42)
2. Missing input validation on email field
3. Plaintext password in log statement (line 78)

## Gemini Findings
1. Authentication bypass via header manipulation
2. SQL injection in user_id (matches GPT)
3. Rate limiting missing on login endpoint

## Consensus Issues
- SQL injection vulnerability (line 42) - CRITICAL

## Divergent Perspectives
- GPT flagged logging issue, Gemini focused on rate limiting
- Both valid, different perspectives on priority

## Recommended Actions
1. Fix SQL injection immediately (use parameterized queries)
2. Add rate limiting to prevent brute force
3. Remove password from logs

Would you like help implementing any of these fixes?
```

## Integration Notes

This skill integrates with the separate `claude-mm-tool` repository:
- Repository: https://github.com/JMacLulich/claude-mm-tool
- Installation: `cd ~/dev/claude-mm-tool && ./run install`
- Command: `ai review --model mm`

## Cost Information

- Typical mm review: $0.02-0.03
- Uses: GPT-5.2 Instant (gpt-5.2-chat-latest) + Gemini-3-flash-preview
- Parallel execution: ~2-3 seconds
- Cached for 24 hours

## Troubleshooting

**"ai: command not found"**
→ Tool not installed. Guide user through installation.

**"Error: OPENAI_API_KEY not set"**
→ API keys not configured. User needs to create `~/.config/claude-mm-tool/env`

**"No changes to review"**
→ Ask user what they want to review (specific files, branch diff, etc.)
