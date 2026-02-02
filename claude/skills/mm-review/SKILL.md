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
  5. "Testing (QA)" - "Focus on test coverage, testability, edge cases, and quality assurance"

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

### Testing (QA) Focus Example

When user selects "Testing", the models should analyze:

- **Test Coverage Gaps**: Missing unit, integration, or E2E tests
- **Testability Issues**: Code that's difficult to test, tight coupling, hidden dependencies
- **Edge Cases**: Boundary conditions, null handling, empty states, overflow scenarios
- **Test Design**: Proper use of test patterns (AAA, Given-When-Then), clear test names
- **Mocking Strategy**: Over-mocking, under-mocking, brittle test doubles
- **Test Smells**: Flaky tests, slow tests, test duplication, assertion quality
- **Property-Based Testing**: Opportunities for generative/invariant testing
- **Error Paths**: Exception handling, failure scenarios, graceful degradation
- **Concurrency**: Race conditions, deadlocks, thread safety in tests
- **Test Pyramid**: Balance between unit/integration/E2E tests
- **Data Driven**: Parameterized tests, test fixtures, test data factories
- **API Testing**: Contract validation, response schemas, status codes
- **Security Testing**: Injection attempts, authentication flows in tests
- **Performance Testing**: Load testing, stress testing, benchmarking gaps
- **Accessibility**: WCAG compliance in UI tests, screen reader testing
- **Internationalization**: Locale handling, timezone, character encoding in tests
- **CI/CD**: Test execution time, parallelization, test result reporting
- **Mutation Testing**: Weak assertions, code coverage vs actual test quality

Example output:
```
User: "get a mm review" → Selects "Testing (QA)"

Claude: Running parallel testing review with GPT + Gemini...
[Executes: git diff | ai review --model mm --focus testing]

# Multi-Model Review Summary

## GPT Testing Findings
1. Missing edge case test for negative values in calculate_discount (line 23)
2. UserService.test_delete_user has over-mocked database layer - tests implementation not behavior
3. No integration tests for payment webhook endpoint (critical business logic)
4. Test suite lacks property-based tests for string validation functions
5. API tests don't validate response headers (Content-Type, Rate-Limit)
6. Flaky test: CacheService.test_expiry (timing-dependent, needs mock clock)
7. No load testing for search endpoint (handles 10x traffic in prod)

## Gemini Testing Findings
1. calculate_discount missing boundary tests (0, max_value, overflow)
2. Test fixtures not reused - test data scattered, maintenance burden
3. No mutation testing coverage - assertions may be passing without validating behavior
4. Missing concurrency tests for OrderService.order_update (potential race conditions)
5. Error path tests incomplete - no tests for database connection failures
6. No accessibility tests in React component suite
7. Integration tests don't clean up database state - cross-test pollution
8. API contract tests missing - schema validation not tested

## Consensus Issues (Both Models Agree)
- Edge case coverage gaps in calculate_discount - HIGH PRIORITY
- Integration test gap for payment webhook - CRITICAL (business logic)
- No load testing for production-critical endpoints - HIGH PRIORITY
- Test data management issues - MEDIUM PRIORITY

## Divergent Perspectives
- GPT focused on over-mocking in unit tests, Gemini identified test fixture duplication
- Both valid: need to balance mock usage with fixture strategy
- GPT flagged flaky tests, Gemini identified test isolation issues (related problems)

## Recommended Actions
1. Add edge case tests to calculate_discount: 0, negative, max, overflow scenarios
2. Create integration test suite for payment webhook with test doubles for external services
3. Implement load testing for search endpoint using k6 or locust (baseline: 1000 RPS)
4. Set up mutation testing (stryker-mutator) to measure test quality
5. Refactor UserService tests to reduce mocking, test public interface not internals
6. Add test data factories/faker.js for consistent test data
7. Add property-based tests for validation functions (fast-check for JS, hypothesis for Python)
8. Fix flaky CacheService.test_expiry by injecting mock clock
9. Add concurrency tests for OrderService with race condition detection
10. Set up test cleanup hooks (beforeEach, afterEach) to prevent state pollution
11. Add API contract tests (pact, openapi-schema-validator) for all endpoints
12. Create accessibility test suite (jest-axe) for React components

Would you like help implementing any of these test improvements?
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
