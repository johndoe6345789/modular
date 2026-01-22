# Compiler Architecture

This document provides an overview of the Mojo compiler architecture.

## Overview

The Mojo compiler is structured as a traditional compiler with distinct phases:

1. **Frontend**: Lexical analysis and parsing
2. **Semantic Analysis**: Type checking and name resolution
3. **IR Generation**: Lowering to MLIR
4. **Optimization**: MLIR optimization passes
5. **Code Generation**: LLVM backend and machine code generation

## Frontend

### Lexer (`frontend/lexer.mojo`)

The lexer tokenizes Mojo source code. Key responsibilities:

- Character-by-character processing
- Token classification (keywords, identifiers, literals, operators)
- Location tracking for error reporting
- Indentation handling (Python-like syntax)

### Parser (`frontend/parser.mojo`)

The parser builds an Abstract Syntax Tree (AST) from tokens. Key responsibilities:

- Recursive descent parsing
- AST construction
- Syntax error reporting
- Support for all Mojo syntax constructs

### Source Location (`frontend/source_location.mojo`)

Tracks source code locations for error reporting and debugging.

## Semantic Analysis

### Type System (`semantic/type_system.mojo`)

Defines the Mojo type system:

- Builtin types (Int, Float64, Bool, String)
- User-defined types (structs)
- Parametric types (generics)
- Trait types
- Reference types (owned, borrowed, mutable)

### Symbol Table (`semantic/symbol_table.mojo`)

Manages name resolution and scoping:

- Variable declarations
- Function and struct definitions
- Scope hierarchy
- Name shadowing

### Type Checker (`semantic/type_checker.mojo`)

Performs semantic analysis:

- Type checking
- Type inference
- Ownership and lifetime checking
- Trait conformance checking

## IR Generation

### MLIR Generator (`ir/mlir_gen.mojo`)

Lowers typed AST to MLIR:

- Converts Mojo constructs to MLIR operations
- Uses Mojo dialect for language-specific operations
- Uses standard MLIR dialects (arith, scf, func) for common operations

### Mojo Dialect (`ir/mojo_dialect.mojo`)

Defines Mojo-specific MLIR dialect:

- Operations: `mojo.func`, `mojo.own`, `mojo.borrow`, `mojo.move`, `mojo.copy`
- Types: `!mojo.value<T>`, `!mojo.ref<T>`, `!mojo.mut_ref<T>`
- Memory model representation

## Optimization

### Optimizer (`codegen/optimizer.mojo`)

Applies optimization passes to MLIR:

- **Level 0**: No optimization
- **Level 1**: Basic optimizations (inlining, constant folding, DCE)
- **Level 2**: Loop optimizations, move elimination
- **Level 3**: Aggressive optimizations (trait devirtualization)

## Code Generation

### LLVM Backend (`codegen/llvm_backend.mojo`)

Generates native code:

- Lowers MLIR to LLVM IR
- Applies LLVM optimization passes
- Generates object files
- Links with runtime libraries

## Runtime Support

### Memory Management (`runtime/memory.mojo`)

Provides memory allocation functions:

- `malloc`, `free`, `realloc`, `calloc`
- Reference counting (if needed)
- Integration with system allocator

### Async Runtime (`runtime/async_runtime.mojo`)

Supports async/await and coroutines:

- Coroutine creation and management
- Task spawning and execution
- Async executor

### Type Reflection (`runtime/reflection.mojo`)

Provides runtime type information:

- Type names
- Type sizes and alignment
- Trait information

## Compilation Pipeline

```
Source Code (.mojo)
    ↓
[Lexer] → Tokens
    ↓
[Parser] → AST
    ↓
[Type Checker] → Typed AST
    ↓
[MLIR Generator] → MLIR (Mojo dialect)
    ↓
[Optimizer] → Optimized MLIR
    ↓
[LLVM Backend] → LLVM IR → Object File
    ↓
[Linker] → Executable
```

## Design Principles

1. **Modularity**: Clean separation between phases
2. **Extensibility**: Easy to add new optimizations and backends
3. **Performance**: Leverage LLVM infrastructure for optimization
4. **Compatibility**: Work with existing stdlib
5. **Error Reporting**: Helpful diagnostics at every phase

## Future Work

- Complete implementation of all phases
- GPU support (CUDA, ROCm, Metal backends)
- Python interoperability
- IDE integration (Language Server Protocol)
- Debugging support (DWARF generation)
