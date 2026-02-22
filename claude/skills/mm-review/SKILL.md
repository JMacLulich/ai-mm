---
name: mm-review
description: Run parallel multi-model AI code reviews using GPT + Gemini + Claude Opus + local Ollama for instant multi-perspective feedback on code changes. Use when user says "mm review", "get a mm review", "multimode review", "run mm review", "review with mm", "get multi-model feedback", or "parallel review".
disable-model-invocation: false
allowed-tools: Bash(git *), Bash(which ai), Bash(ai review *), AskUserQuestion
---

# Multi-Model Review Skill

## Purpose

Run parallel multi-model AI code reviews using the `ai-mm`. Provides instant perspectives from GPT + Gemini + Claude Opus for broader feedback.

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

- `ai-mm` must be installed at `~/dev/ai-mm` or `~/../ai-mm`
- API keys configured in `~/.config/ai-mm/env`

## Instructions

### Step 1: Verify Tool Installation

Check if ai-mm is installed:

```bash
which ai
```

If not found, guide user to install:
```bash
cd ~/dev/ai-mm
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
- Claude Opus 4.5 review results
- Displayed side-by-side for comparison

### Step 5: Summarize Results

Present a structured summary:

```markdown
# Multi-Model Review Summary

## GPT Findings
[Summarize GPT's key points - top 3-5 issues]

## Gemini Findings
[Summarize Gemini's key points - top 3-5 issues]

## Claude Opus 4.5 Findings
[Summarize Opus's key points - top 3-5 issues]

## Consensus Issues
[Issues all 3 models agree on - highest priority]

## Partial Consensus (2/3 Models)
[Issues where 2 models agree - high priority]

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

Claude: Running parallel security review with GPT + Gemini + Claude Opus 4.5...
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

## Claude Opus 4.5 Findings
1. CSRF token missing on state-changing operations
2. SQL injection in user_id (matches GPT and Gemini)
3. Weak password hashing algorithm (bcrypt recommended over md5)

## Consensus Issues (All 3 Models Agree)
- SQL injection vulnerability (line 42) - CRITICAL

## Partial Consensus (2/3 Models Agree)
- Rate limiting missing on login endpoint (GPT + Gemini)
- Password security concerns (GPT logging issue + Opus hashing recommendation)

## Divergent Perspectives
- Opus identified CSRF risk that others missed
- GPT and Gemini focused on input validation, Opus focused on authentication tokens
- All valid, different security perspectives

## Recommended Actions
1. Fix SQL injection immediately (use parameterized queries) - **ALL MODELS AGREE**
2. Add CSRF tokens to all state-changing forms - **Opus priority**
3. Upgrade password hashing from md5 to bcrypt - **Opus priority**
4. Add rate limiting to prevent brute force - **GPT + Gemini priority**
5. Remove password from logs - **GPT priority**

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

Claude: Running parallel testing review with GPT + Gemini + Claude Opus 4.5...
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

## Claude Opus 4.5 Testing Findings
1. calculate_discount missing boundary tests (matches Gemini)
2. No golden master / snapshot tests for UI components
3. Test database not isolated - integration tests interfere with local dev data
4. Missing chaos engineering / failure injection tests
5. Test assertions too brittle - exact string matching instead of structural validation
6. No A/B testing framework for validating business logic changes
7. Performance regression tests absent - no baseline comparisons

## Consensus Issues (All 3 Models Agree)
- Edge case coverage gaps in calculate_discount - HIGH PRIORITY
- Integration test gap for payment webhook - CRITICAL (business logic)
- No load testing for production-critical endpoints - HIGH PRIORITY
- Test data management issues - MEDIUM PRIORITY

## Partial Consensus (2/3 Models Agree)
- Test isolation problems (Gemini + Opus)
- Brittle test patterns (GPT over-mocking + Opus assertion brittleness)

## Divergent Perspectives
- Opus identified golden master / snapshot testing need
- Opus emphasized chaos engineering and failure injection
- GPT focused on mock usage, Gemini on fixtures, Opus on assertions
- All valid, complementary testing perspectives

## Recommended Actions
1. Add edge case tests to calculate_discount: 0, negative, max, overflow scenarios
2. Create integration test suite for payment webhook with test doubles for external services
3. Implement load testing for search endpoint using k6 or locust (baseline: 1000 RPS) - **ALL MODELS AGREE**
4. Set up mutation testing (stryker-mutator) to measure test quality
5. Refactor UserService tests to reduce mocking, test public interface not internals
6. Add test data factories/faker.js for consistent test data
7. Add property-based tests for validation functions (fast-check for JS, hypothesis for Python)
8. Fix flaky CacheService.test_expiry by injecting mock clock
9. Add concurrency tests for OrderService with race condition detection
10. Set up test cleanup hooks (beforeEach, afterEach) to prevent state pollution
11. Add API contract tests (pact, openapi-schema-validator) for all endpoints
12. Create accessibility test suite (jest-axe) for React components
13. Add snapshot testing for UI components - **Opus priority**
14. Implement test database isolation (Docker or separate schema) - **Opus priority**
15. Set up chaos engineering tests (using Chaos Monkey or similar) - **Opus priority**
16. Refactor brittle assertions to use structural validation - **Opus priority**

Would you like help implementing any of these test improvements?
```

## Integration Notes

This skill integrates with the separate `ai-mm` repository:
- Repository: https://github.com/JMacLulich/ai-mm
- Installation: `cd ~/dev/ai-mm && ./run install`
- Command: `ai review --model mm`

## Cost Information

- Typical mm review: $0.05-0.12
- Uses: GPT-5.2 Instant (gpt-5.2-chat-latest) + Gemini-3-flash-preview + Claude Opus 4.5
- Parallel execution: ~2-3 seconds
- Cached for 24 hours

### Model-Specific Reviews

You can also use specific models:
- `ai review --model gpt` - GPT-5.2 Chat Latest only ($1.75/$14 per 1M tokens)
- `ai review --model gemini` - Gemini 3 Flash Preview only ($0.075/$0.30 per 1M tokens)
- `ai review --model claude` - Claude Sonnet 4.5 only ($3/$15 per 1M tokens)
- `ai review --model opus` - Claude Opus 4.5 only ($5/$25 per 1M tokens)
- `ai review --model all` - All 3 fast models (GPT + Gemini + Haiku 4.5) in parallel

## Troubleshooting

**"ai: command not found"**
→ Tool not installed. Guide user through installation.

**"Error: OPENAI_API_KEY not set"**
→ API keys not configured. User needs to create `~/.config/ai-mm/env`

**"No changes to review"**
→ Ask user what they want to review (specific files, branch diff, etc.)
