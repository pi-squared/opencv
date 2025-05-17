# Reproducibility Test for createsamples

This directory contains tests to verify that the createsamples tool produces reproducible results when the same RNG seed is used.

## Files
- `test_reproducible.cpp` - Tests basic RNG seed reproducibility
- `test_createsamples_reproduce.cpp` - Tests createsamples reproducibility

## How to Test

1. Compile OpenCV with the fixes
2. Run the test:
   ```
   ./test_createsamples_reproduce 12345 12345
   ```
   
   This should show "SUCCESS: Samples are reproducible with the same seed!"

3. To verify it fails without the fix:
   ```
   ./test_createsamples_reproduce 12345 54321
   ```
   
   This should show "FAILURE: Samples are not reproducible with the same seed!"

## Fix Description

The fixes ensure proper seeding of all random number generation in the createsamples tool, making the results reproducible when the same RNG seed is provided through the -rngseed parameter.