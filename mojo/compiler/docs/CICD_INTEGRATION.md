# CI/CD Integration Guide

**Last Updated**: January 22, 2026  
**Status**: Complete ✅

## Overview

This document describes the CI/CD integration for the Mojo compiler, including test infrastructure, build system configuration, and continuous integration setup.

## Test Infrastructure

### Test Suite Organization

The compiler includes 14 comprehensive test targets organized by component and phase:

#### Core Component Tests
- `test_lexer` - Tokenization and lexical analysis
- `test_type_checker` - Type checking and semantic analysis
- `test_mlir_gen` - MLIR code generation
- `test_backend` - LLVM backend and compilation
- `test_compiler_pipeline` - End-to-end pipeline integration

#### Phase 2 Tests (Control Flow & Structs)
- `test_operators` - Arithmetic, comparison, and boolean operators
- `test_control_flow` - If/elif/else, while, for loops
- `test_structs` - Struct definitions and methods
- `test_phase2_structs` - Advanced struct features

#### Phase 3 Tests (Traits & Codegen)
- `test_phase3_traits` - Trait definitions and conformance
- `test_phase3_iteration` - Enhanced collection iteration

#### Phase 4 Tests (Advanced Features)
- `test_phase4_generics` - Parametric types and generics
- `test_phase4_inference` - Type inference
- `test_phase4_ownership` - Ownership system and borrow checking

#### Integration Tests
- `test_end_to_end` - Full compilation to native executables (manual)

### Test Suite Target

A convenience target is provided to run all tests:

```bash
./bazelw test //mojo/compiler:compiler_tests
```

This runs all component and phase tests (excluding manual tests).

## Bazel Configuration

### BUILD.bazel Structure

The compiler's `BUILD.bazel` file defines:

```python
load("//bazel:api.bzl", "mojo_library", "mojo_test")

# Test suite aggregation
test_suite(
    name = "compiler_tests",
    tests = [":test_lexer", ":test_type_checker", ...],
)

# Individual test targets
mojo_test(
    name = "test_lexer",
    srcs = ["test_lexer.mojo"],
    deps = ["//mojo/compiler/src:frontend"],
)
```

### Test Tags

Tests use tags for CI/CD filtering:

- `manual` - Must be run explicitly (not included in `//...`)
- `requires-llvm-tools` - Requires external LLVM toolchain
- `requires-network` - Needs network access (filtered in CI)

## CI/CD Pipeline

### GitHub Actions Integration

The compiler tests are integrated with the existing `.github/workflows/build_and_test.yml` workflow:

```yaml
- name: Build and Test
  run: |
    BB_DISABLE_SIDECAR=1 ./bazelw test \
      --config=ci \
      --config=public-cache \
      --build_tag_filters=-skip-external-ci-${{ matrix.os }},-requires-network \
      --test_tag_filters=-skip-external-ci-${{ matrix.os }},-requires-network \
      -- //...
```

### CI Configuration Details

- **Platforms**: `large-oss-linux` runners
- **Test Filters**: Excludes `skip-external-ci-*` and `requires-network` tags
- **Build Filters**: Same filtering applied to build targets
- **Manual Tests**: Tests tagged `manual` must be run explicitly
- **Caching**: BuildBuddy remote cache for faster builds

### Automatic Execution

Tests run automatically on:
- Pull requests (opened, synchronized, reopened)
- Pushes to `main` branch
- Manual workflow dispatch

### Test Exclusions in CI

The following tests are excluded from automated CI:
- `test_end_to_end` - Requires LLVM tools (`llc`, `cc`) not available in CI

## Local Development

### Running Tests Locally

#### Run All Tests
```bash
./bazelw test //mojo/compiler:compiler_tests
```

#### Run Specific Component
```bash
./bazelw test //mojo/compiler:test_lexer
./bazelw test //mojo/compiler:test_type_checker
./bazelw test //mojo/compiler:test_mlir_gen
```

#### Run Phase-Specific Tests
```bash
./bazelw test //mojo/compiler:test_phase2_structs
./bazelw test //mojo/compiler:test_phase3_traits
./bazelw test //mojo/compiler:test_phase4_generics
```

#### Run Manual Tests
```bash
# Requires LLVM tools (llc, cc)
./bazelw test //mojo/compiler:test_end_to_end
```

### Alternative: Direct Execution

Tests can also be run directly with Mojo:

```bash
cd mojo/compiler
mojo test_lexer.mojo
mojo test_type_checker.mojo
```

This is useful for:
- Quick iteration during development
- Debugging specific tests
- Running tests without Bazel

## Build Artifacts

### Runtime Library

The C runtime library must be built before running end-to-end tests:

```bash
cd mojo/compiler/runtime
make
```

This creates `libmojo_runtime.a` which is linked with compiled programs.

### Ignored Artifacts

The following build artifacts are excluded from version control:

```gitignore
mojo/compiler/runtime/*.o
mojo/compiler/runtime/*.a
```

These are generated during builds and should not be committed.

## Test Coverage

### Component Coverage

| Component | Test Files | Coverage |
|-----------|-----------|----------|
| Lexer | test_lexer.mojo | Tokenization, all token types |
| Parser | (in component tests) | AST construction |
| Type Checker | test_type_checker.mojo | Type validation, inference |
| MLIR Gen | test_mlir_gen.mojo | IR generation |
| Backend | test_backend.mojo | LLVM IR, compilation |
| Pipeline | test_compiler_pipeline.mojo | Integration |

### Feature Coverage

| Phase | Features | Test Files |
|-------|----------|-----------|
| Phase 1 | Basic compilation | All component tests |
| Phase 2 | Control flow, structs | test_operators, test_control_flow, test_structs |
| Phase 3 | Traits, iteration | test_phase3_traits, test_phase3_iteration |
| Phase 4 | Generics, inference | test_phase4_generics, test_phase4_inference, test_phase4_ownership |

### Integration Coverage

- **Compiler Pipeline**: test_compiler_pipeline.mojo
- **End-to-End**: test_end_to_end.mojo (requires LLVM tools)

## Troubleshooting

### Test Failures

#### Build Errors
```bash
# Check BUILD.bazel syntax
./bazelw query //mojo/compiler:all

# Verify dependencies
./bazelw query "deps(//mojo/compiler:test_lexer)"
```

#### Import Errors
Ensure all dependencies are properly declared in `BUILD.bazel`:
```python
deps = [
    "//mojo/compiler/src:frontend",
    "//mojo/compiler/src:semantic",
]
```

#### LLVM Tool Requirements
End-to-end tests require:
```bash
# Install on Ubuntu/Debian
sudo apt-get install llvm gcc

# Verify installation
which llc  # LLVM compiler
which cc   # C compiler
```

### CI Issues

#### Tests Not Running
- Check test tags don't include filtered tags
- Verify test is included in test_suite
- Check platform compatibility

#### Timeout Issues
Add custom timeout to test target:
```python
mojo_test(
    name = "slow_test",
    timeout = "long",  # short, moderate, long, eternal
    ...
)
```

## Contributing

### Adding New Tests

1. Create test file: `test_new_feature.mojo`
2. Add test target to `BUILD.bazel`:
   ```python
   mojo_test(
       name = "test_new_feature",
       srcs = ["test_new_feature.mojo"],
       deps = ["//mojo/compiler/src:component"],
   )
   ```
3. Add to test suite:
   ```python
   test_suite(
       name = "compiler_tests",
       tests = [..., ":test_new_feature"],
   )
   ```

### Test Guidelines

- **Naming**: Use `test_*.mojo` pattern
- **Organization**: Group related tests by component or phase
- **Dependencies**: Declare minimal required dependencies
- **Tags**: Use appropriate tags for special requirements
- **Documentation**: Add comments explaining test purpose

## Future Enhancements

### Planned Improvements

- [ ] Test coverage reporting
- [ ] Performance benchmarking in CI
- [ ] Cross-platform testing (macOS, Windows)
- [ ] Nightly builds with extended tests
- [ ] Integration with external test frameworks

### Test Expansion

- [ ] More edge case coverage
- [ ] Negative test cases (error handling)
- [ ] Performance regression tests
- [ ] Memory safety tests
- [ ] Concurrency tests

## References

- [GitHub Actions Workflow](/.github/workflows/build_and_test.yml)
- [Bazel Documentation](https://bazel.build/)
- [Compiler README](/mojo/compiler/README.md)
- [Developer Guide](/mojo/compiler/DEVELOPER_GUIDE.md)

---

**Document Version**: 1.0  
**Last Updated**: January 22, 2026  
**Status**: Complete ✅
