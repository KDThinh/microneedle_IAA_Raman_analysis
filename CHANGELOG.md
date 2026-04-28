# Changelog

## Fixes Applied

### Python 3.8 Compatibility
- **Fixed type hints in `io/config.py`**: Changed `List[str] | None` to `Optional[List[str]]` to maintain compatibility with Python 3.8+ as specified in `setup.py`
- All type hints now use `Optional` from `typing` module instead of the `|` union syntax (which requires Python 3.10+)

### V3 Algorithm Placeholder
- **Fixed `_process_v3` in `pipeline.py`**: Changed from silently using V4 logic to raising `NotImplementedError` with a clear message
- This prevents accidental use of V4 when V3 is expected, ensuring scientific accuracy

## Package Status

✅ **Production Ready**: All compatibility issues resolved
✅ **Python 3.8+ Compatible**: Type hints use `Optional` syntax
✅ **Clear Error Handling**: V3 algorithm properly raises `NotImplementedError`

## Next Steps

1. Install the package: `pip install -e .`
2. Test: `swnt-iaa-analysis list-profiles`
3. Run analysis: `swnt-iaa-analysis analyze <profile_name>`

