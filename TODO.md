# TODO

## Test Fixes Needed

The test suite has been scaffolded but needs fixes:

### test_costs.py
- `test_format_cost_warning` - Function signature doesn't match implementation
- `test_should_warn_about_cost` - Function signature doesn't match implementation
- `test_estimate_cost_from_text` - Missing pricing data for model

### test_cache.py
- `test_cache_response_and_get` - Cache directory path issue
- `test_cache_expiry` - Cache TTL logic doesn't match test expectations
- `test_clear_cache` - Return value doesn't match expectations
- `test_get_cache_stats` - Missing 'total_size_bytes' key in stats

### test_retry.py
- All tests passing ✅

## Next Steps

1. Fix test API mismatches
2. Ensure cache tests use proper temp directories
3. Add integration tests for actual API calls (with mocking)
4. Add tests for providers (openai, google, anthropic)
5. Add tests for api.py module

## Status

- ✅ Project structure created
- ✅ ./run commands working (lint, test, install)
- ✅ Carapace completions added
- ✅ Installation working
- ⚠️  Tests scaffolded but need fixes (8 passing, 7 failing)
- ✅ Integrated as skill in system-playbooks
