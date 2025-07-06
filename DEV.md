# Development Guide

## Setup Auto Formatting

This project uses automatic code formatting before each commit. The formatting is enforced through Git hooks.

First install `clang-format` which is required by the formatter script(`formatter.sh`):

```bash
sudo apt-get install clang-format
```

Then set up the Git hooks, run:

```bash
./scripts/setup-git-hooks.sh
```

This will create a symlink from `.git/hooks/pre-commit` to `scripts/pre-commit`, which runs the formatter script (`formatter.sh`) before each commit and automatically stages any changes made by the formatter.

## Build and Tests

```bash
bash scripts/build.sh
``` 
Check more options with it's helper information with `-h`.

## Run Tests

Re-build and run all tests:
```bash
bash scripts/run_tests.sh
```

Or run specific tests with cpp only build:
```bash
bash scripts/run_tests.sh --cpp-only camera/model util/se3 
```
