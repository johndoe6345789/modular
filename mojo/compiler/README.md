# Open Source Mojo Compiler

This directory contains the implementation of the open source Mojo compiler as outlined in [the compiler proposal](../proposals/open-source-compiler.md).

## Overview

The Mojo compiler is a from-scratch implementation that compiles Mojo source code to native executables. It is built on MLIR and LLVM infrastructure and designed to work seamlessly with the open source Mojo standard library.

## Architecture

The compiler consists of several key components:

```
┌─────────────────────────────────────────────────────────────┐
│                     Mojo Source Code                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Frontend (Parser + Sema)                    │
│  • Lexer: Tokenize Mojo source                              │
│  • Parser: Build AST from tokens                            │
│  • Semantic Analysis: Type checking, name resolution        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 IR Generation (to MLIR)                      │
│  • Lower Mojo AST to MLIR dialects                          │
│  • Mojo-specific MLIR dialects                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              MLIR Optimization Pipeline                      │
│  • High-level optimizations                                 │
│  • Target-independent transformations                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 Backend Code Generation                      │
│  • Lower MLIR to LLVM IR                                    │
│  • Target-specific optimizations                            │
│  • Machine code generation                                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Native Executable / Library                     │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
mojo/compiler/
├── src/                      # Source code
│   ├── frontend/             # Lexer and parser
│   ├── semantic/             # Type checking and semantic analysis
│   ├── ir/                   # MLIR dialect definitions and IR generation
│   ├── codegen/              # Code generation and optimization
│   ├── runtime/              # Compiler runtime support
│   └── utils/                # Utility functions and helpers
├── test/                     # Test suite
│   ├── frontend/             # Frontend tests
│   ├── semantic/             # Semantic analysis tests
│   ├── ir/                   # IR generation tests
│   ├── codegen/              # Codegen tests
│   └── runtime/              # Runtime tests
├── docs/                     # Documentation
├── examples/                 # Example programs
└── README.md                 # This file
```

## Components

### Frontend (Lexer and Parser)

**Location**: `src/frontend/`

Responsible for:
- Tokenizing Mojo source code
- Building Abstract Syntax Tree (AST)
- Reporting syntax errors with helpful diagnostics

Key features:
- Support for all Mojo syntax (struct, fn, var, def, etc.)
- Parameter blocks `[T: Type]`
- Decorators (`@value`, `@register_passable`, etc.)
- Python interop syntax

### Semantic Analysis

**Location**: `src/semantic/`

Responsible for:
- Type checking and inference
- Name resolution and scoping
- Trait resolution
- Lifetime and ownership analysis
- Compile-time evaluation

Key features:
- Parametric type system
- Trait-based generics
- Value semantics and ownership checking
- Reference lifetime validation

### IR Generation

**Location**: `src/ir/`

Responsible for:
- Lowering Mojo AST to MLIR
- Defining Mojo-specific MLIR dialects
- Memory model operations (own, borrow, move, copy)

Key dialects:
- `mojo` dialect: Core Mojo operations
- Integration with standard MLIR dialects (arith, scf, func, cf, llvm)

### Code Generation

**Location**: `src/codegen/`

Responsible for:
- MLIR optimization pipeline
- Lowering to LLVM IR
- Target-specific optimizations
- Machine code generation

Optimizations:
- Inlining, constant folding, DCE
- Loop optimizations
- Move/copy elimination
- Trait devirtualization

### Runtime Support

**Location**: `src/runtime/`

Provides runtime support for:
- Memory management (malloc, free, realloc)
- Async/coroutine runtime
- Type reflection
- String and collection operations
- C library interoperability
- Python interoperability

## Building

To build the compiler:

```bash
# From repository root
./bazelw build //mojo/compiler/...

# Run tests
./bazelw test //mojo/compiler/...
```

## Usage

```bash
# Compile a Mojo file
mojo-oss build myprogram.mojo

# Compile with options
mojo-oss build --target=x86_64-linux \
              --stdlib-path=/path/to/stdlib \
              --opt-level=3 \
              myprogram.mojo

# Run tests
mojo-oss test ./test/
```

## Implementation Status

### Phase 1: Minimal Viable Compiler (Current)
- [ ] Lexer and parser for basic Mojo syntax
- [ ] Type checker for simple types
- [ ] MLIR code generation for basic operations
- [ ] LLVM backend integration
- [ ] Compile and run "Hello, World!"

### Phase 2: Core Language Features
- [ ] Full type system (parametrics, traits)
- [ ] Ownership and lifetime checking
- [ ] Complete control flow (if, while, for)
- [ ] Struct definitions and methods
- [ ] Compile basic stdlib modules

### Phase 3: Advanced Features
- [ ] Python interop
- [ ] Async/await
- [ ] GPU support
- [ ] Compile entire stdlib
- [ ] Optimization pipeline

### Phase 4: Production Ready
- [ ] Performance parity with existing compiler
- [ ] Complete language spec coverage
- [ ] Comprehensive error messages
- [ ] IDE integration (LSP)
- [ ] Debugging support (DWARF)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

Key areas where we need help:
- Parser and lexer implementation
- Type system design and implementation
- MLIR dialect definitions
- Optimization passes
- Testing and documentation

## Design Documents

- [Open Source Compiler Proposal](../proposals/open-source-compiler.md) - The full design specification

## Goals

1. **Full Language Support**: Implement complete Mojo language specification
2. **Standard Library Compatibility**: Work seamlessly with the existing open source stdlib
3. **Performance**: Achieve competitive performance with existing implementation
4. **C Library Interoperability**: Preserve seamless integration with C libraries
5. **Modularity**: Clean separation of concerns for maintainability
6. **Extensibility**: Easy to add new backends and optimizations

## Technical Challenges

1. **Parametric Type System**: Compile-time evaluation of type parameters
2. **Ownership and Lifetimes**: Proving safety without explicit annotations
3. **MLIR Dialect Design**: Efficiently representing Mojo semantics
4. **Standard Library ABI**: Maintaining compatibility with existing stdlib
5. **Performance**: Matching performance of highly optimized existing compiler

## License

Licensed under the Apache License v2.0 with LLVM Exceptions.
See [LICENSE](../../LICENSE) for details.

## References

- [LLVM Project](https://llvm.org/)
- [MLIR Documentation](https://mlir.llvm.org/)
- [Mojo Standard Library](../stdlib/)
- [Mojo Language Manual](https://docs.modular.com/mojo/manual/)
