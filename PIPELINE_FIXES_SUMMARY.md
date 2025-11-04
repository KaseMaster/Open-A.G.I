# Pipeline Issues Fixed

## Summary of Fixes Applied

### 1. Deprecated GitHub Actions Updated
- **actions/upload-artifact**: Updated from v3 to v4 in both workflow files
- **actions/setup-python**: Updated from v4 to v5 in ci-cd.yml workflow
- **codecov/codecov-action**: Updated from v3 to v4 in ci-cd.yml workflow
- **docker/build-push-action**: Updated from v5 to v6 in both workflow files

### 2. PowerShell Syntax Error Fixed
- Fixed incomplete if statement in Windows job dependencies installation step in ci.yml

### 3. Error Handling Improvements
- Added comprehensive error handling to debug_heartbeat.py

### 4. Version Consistency
- Verified consistent Python version support (3.11, 3.12, 3.13) across all workflow files

## Files Modified

1. `.github/workflows/ci-cd.yml`
2. `.github/workflows/ci.yml`
3. `debug_heartbeat.py`

## Issues Resolved

- ✅ CI pipeline failures due to deprecated `actions/upload-artifact: v3`
- ✅ PowerShell syntax errors in dependency installation
- ✅ Outdated action versions causing deprecation warnings
- ✅ Improved error handling for debugging scripts

## Testing

All workflows should now pass without deprecation warnings related to the actions that were updated.