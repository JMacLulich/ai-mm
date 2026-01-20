# Claude MM Tool - Project Instructions

## Code Quality

### Linting

**Always run `./run lint fix` before committing code.**

This project uses Ruff for linting with a 100-character line limit. Auto-fix handles most issues:

```bash
./run lint fix    # Auto-fix linting issues
./run lint        # Check for remaining issues
```

**When to run:**
- Before every commit
- After writing/modifying Python code
- As part of the push workflow

## Git Commit Guidelines

**IMPORTANT: Do NOT include "Co-Authored-By" lines in commit messages.**

When creating commits, write clear, concise commit messages that focus on what changed and why, without attribution lines.

### Before Every Commit

1. **Run linting**: `./run lint fix` to auto-fix issues
2. **Verify**: `./run lint` to check for remaining errors
3. **Fix manually**: Address any errors that couldn't be auto-fixed
4. **Stage changes**: Include any files modified by lint fix
5. **Commit**: Create the commit

## Push Workflow

When the user says "push", follow this workflow in order:

1. **Auto-fix lint**: Run `./run lint fix` to auto-fix any linting issues
2. **Manually fix remaining lint errors**: If `./run lint fix` couldn't fix all errors, manually fix them
3. **Verify lint**: Run `./run lint` to confirm all linting errors are resolved
4. **Fix test failures**: Run `./run test` and fix any failing tests
5. **Push**: Run `git push` to push commits to remote

**Important:**
- Always run `./run lint fix` first before manual fixes
- Do not push if linting errors remain
- Do not push if tests fail
