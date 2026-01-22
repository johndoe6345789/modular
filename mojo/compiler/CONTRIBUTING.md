# Contributing to the Mojo Compiler

Thank you for your interest in contributing to the open source Mojo compiler!

## Getting Started

1. **Read the Proposal**: Start with [open-source-compiler.md](../../proposals/open-source-compiler.md) to understand the overall design.

2. **Review the Architecture**: Read [architecture.md](../docs/architecture.md) to understand how the compiler is structured.

3. **Explore the Code**: Browse the source code in `src/` to see the current implementation.

## Current Status

The compiler is in the **Phase 1: Foundation** stage. Most modules contain:
- Comprehensive documentation
- Type definitions and interfaces
- Stub implementations with TODO markers
- Test directory structure (tests to be added)

## Areas Where You Can Help

### High Priority

1. **Frontend Implementation**
   - Complete the lexer (tokenization logic)
   - Complete the parser (AST construction)
   - Add comprehensive error reporting

2. **Type System**
   - Implement builtin types
   - Implement parametric types
   - Implement trait system

3. **Testing**
   - Add unit tests for lexer
   - Add unit tests for parser
   - Add integration tests

### Medium Priority

4. **MLIR Integration**
   - Define complete Mojo dialect
   - Implement IR generation
   - Add MLIR optimization passes

5. **Code Generation**
   - Implement LLVM backend
   - Add optimization passes
   - Support multiple targets

### Future Work

6. **Advanced Features**
   - Python interoperability
   - Async/await support
   - GPU support (CUDA, ROCm, Metal)

7. **Tooling**
   - Language Server Protocol (LSP) implementation
   - Debugger support (DWARF)
   - Build system integration

## How to Contribute

### Finding an Issue

Look for files with TODO comments or modules that are mostly stubs. Good starting points:

```bash
# Find all TODO comments
grep -r "TODO" mojo/compiler/src/

# Start with the lexer - it's relatively self-contained
mojo/compiler/src/frontend/lexer.mojo

# Or the type system
mojo/compiler/src/semantic/type_system.mojo
```

### Making Changes

1. **Keep changes focused**: One feature or fix per pull request
2. **Add tests**: Every feature should have tests
3. **Document your code**: Add docstrings and comments
4. **Follow the style**: Match the existing code style
5. **Sign your commits**: Use `git commit -s`

### Code Style Guidelines

- Use the Mojo formatter: `mojo format`
- Add docstrings to all public APIs
- Include type annotations
- Keep functions small and focused
- Use descriptive variable names

### Testing

When adding tests:

```bash
# Place tests in the corresponding test directory
mojo/compiler/test/frontend/  # for frontend tests
mojo/compiler/test/semantic/  # for semantic tests
# etc.

# Tests should use the existing test infrastructure
# See mojo/stdlib/test/ for examples
```

### Pull Request Process

1. Create a branch with a descriptive name
2. Make your changes
3. Add tests
4. Run existing tests (once implemented)
5. Update documentation if needed
6. Submit a pull request with:
   - Clear description of changes
   - Motivation for the change
   - Any breaking changes noted

## Development Setup

### Prerequisites

- Mojo SDK (nightly build recommended)
- LLVM/MLIR (for backend development)
- Bazel build system
- Git

### Building

```bash
# Build the compiler
./bazelw build //mojo/compiler/...

# Run tests
./bazelw test //mojo/compiler/...
```

## Architecture Overview

Quick overview of the compiler pipeline:

```
Source Code → [Lexer] → Tokens
           → [Parser] → AST
           → [Type Checker] → Typed AST
           → [MLIR Gen] → MLIR
           → [Optimizer] → Optimized MLIR
           → [LLVM Backend] → Native Code
```

Each phase is in its own directory under `src/`.

## Communication

- **GitHub Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions
- **Discussions**: Design discussions and questions

## Resources

- [LLVM Documentation](https://llvm.org/docs/)
- [MLIR Documentation](https://mlir.llvm.org/)
- [Mojo Language Manual](https://docs.modular.com/mojo/manual/)
- [Compiler Design Resources](https://en.wikipedia.org/wiki/Compilers:_Principles,_Techniques,_and_Tools)

## Code of Conduct

Please follow the [Code of Conduct](../../CODE_OF_CONDUCT.md) in all interactions.

## License

All contributions will be licensed under the Apache License v2.0 with LLVM Exceptions.
See [LICENSE](../../LICENSE) for details.

## Questions?

If you have questions:
1. Check the documentation in `docs/`
2. Look for similar code in the codebase
3. Open a GitHub Discussion
4. Ask in the pull request if it's specific to your change

Thank you for contributing to making Mojo's compiler open source!
