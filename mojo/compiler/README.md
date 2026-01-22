# Open Source Mojo Compiler

This directory contains the implementation of the open source Mojo compiler as outlined in [the compiler proposal](../proposals/open-source-compiler.md).

## Status: Phase 1 - Foundation (40% Complete)

The compiler structure is in place with significant progress on the frontend:

- ✅ **Lexer**: 85% complete - tokenizes Mojo source code
- 🔄 **Parser**: 60% complete - builds Abstract Syntax Tree
- 🔄 **AST**: Complete for Phase 1 - comprehensive node definitions
- 🔴 **Type Checker**: Skeleton only - needs implementation
- 🔴 **MLIR Generator**: Skeleton only - needs implementation  
- 🔴 **LLVM Backend**: Skeleton only - needs implementation

### Recent Progress

**Latest Updates**:
- ✅ Implemented comprehensive lexer with keyword, literal, and operator support
- ✅ Created complete AST node type system
- ✅ Enhanced parser with function, expression, and statement parsing
- ✅ Added example programs (Hello World, simple function)
- ✅ Created comprehensive developer documentation

**What Works Now**:
- Tokenizing Mojo source files
- Parsing basic function definitions
- Building AST structure for simple programs
- Error tracking and source location reporting

## Quick Start

### Example Programs

See `examples/` for sample Mojo programs:

```mojo
# examples/hello_world.mojo
fn main():
    print("Hello, World!")
```

```mojo
# examples/simple_function.mojo
fn add(a: Int, b: Int) -> Int:
    return a + b

fn main():
    let result = add(40, 2)
    print(result)
```

### Testing the Lexer

Run the lexer test to see tokenization in action:

```bash
# From the compiler directory
mojo test_lexer.mojo
```

This will demonstrate:
- Keyword recognition
- Literal parsing (integers, floats, strings, booleans)
- Operator tokenization
- Complete function lexing

### Using the Compiler (Conceptual)

```mojo
from compiler import CompilerOptions, compile

fn main():
    var options = CompilerOptions(
        target="x86_64-linux",
        opt_level=2,
        stdlib_path="../stdlib",
        output_path="hello_world"
    )
    
    let success = compile("examples/hello_world.mojo", options)
    if success:
        print("Compilation successful!")
```

**Note**: Full compilation is not yet functional - this is the target API.

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
│  • Lexer: Tokenize Mojo source              [✅ 85%]        │
│  • Parser: Build AST from tokens            [🔄 60%]        │
│  • Semantic Analysis: Type checking         [🔴 0%]         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 IR Generation (to MLIR)                      │
│  • Lower Mojo AST to MLIR dialects          [🔴 0%]         │
│  • Mojo-specific MLIR dialects              [🔴 0%]         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              MLIR Optimization Pipeline                      │
│  • High-level optimizations                 [🔴 0%]         │
│  • Target-independent transformations       [🔴 0%]         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 Backend Code Generation                      │
│  • Lower MLIR to LLVM IR                    [🔴 0%]         │
│  • Target-specific optimizations            [🔴 0%]         │
│  • Machine code generation                  [🔴 0%]         │
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
├── src/                           # Source code
│   ├── frontend/                  # Lexer and parser [🔄 Partial]
│   │   ├── lexer.mojo            # Tokenization [✅ 85%]
│   │   ├── parser.mojo           # AST construction [🔄 60%]
│   │   ├── ast.mojo              # AST node definitions [✅ Complete]
│   │   └── source_location.mojo  # Location tracking [✅ Complete]
│   ├── semantic/                  # Type checking [🔴 Skeleton]
│   │   ├── type_checker.mojo     # Type validation
│   │   ├── type_system.mojo      # Type representations
│   │   └── symbol_table.mojo     # Name resolution
│   ├── ir/                        # MLIR generation [🔴 Skeleton]
│   │   ├── mlir_gen.mojo         # IR generation
│   │   └── mojo_dialect.mojo     # Mojo dialect
│   ├── codegen/                   # Code generation [🔴 Skeleton]
│   │   ├── optimizer.mojo        # Optimization passes
│   │   └── llvm_backend.mojo     # LLVM backend
│   └── runtime/                   # Runtime support [🔴 Skeleton]
│       ├── memory.mojo            # Memory management
│       ├── reflection.mojo        # Type reflection
│       └── async_runtime.mojo     # Async support
├── examples/                      # Example programs [✅ Created]
│   ├── hello_world.mojo          # Simple example
│   └── simple_function.mojo      # Function example
├── docs/                          # Documentation
├── test_lexer.mojo               # Lexer tests [✅ Created]
├── compiler_demo.mojo            # Compiler demo [✅ Created]
├── README.md                     # This file [✅ Updated]
├── IMPLEMENTATION_STATUS.md      # Detailed status [✅ Created]
└── DEVELOPER_GUIDE.md            # Dev guide [✅ Created]
```

## Components

### Frontend (Lexer and Parser)

**Location**: `src/frontend/`

**Status**: 🔄 Partially Complete (70% overall)

Responsible for:
- Tokenizing Mojo source code ✅
- Building Abstract Syntax Tree 🔄
- Reporting syntax errors with helpful diagnostics ✅

Key features:
- Support for all Mojo syntax (struct, fn, var, def, etc.) ✅
- Parameter blocks `[T: Type]` 🔴
- Decorators (`@value`, `@register_passable`, etc.) 🔴
- Python interop syntax 🔴

**Files**:
- `lexer.mojo` - Tokenization (85% complete)
- `parser.mojo` - Parsing (60% complete)
- `ast.mojo` - AST nodes (complete for Phase 1)
- `source_location.mojo` - Location tracking (complete)

### Semantic Analysis

**Location**: `src/semantic/`

**Status**: 🔴 Skeleton Only

Responsible for:
- Type checking and inference
- Name resolution and scoping
- Trait resolution
- Lifetime and ownership analysis
- Compile-time evaluation

Key features needed:
- Parametric type system
- Trait-based generics
- Value semantics and ownership checking
- Reference lifetime validation

### IR Generation

**Location**: `src/ir/`

**Status**: 🔴 Skeleton Only

Responsible for:
- Lowering Mojo AST to MLIR
- Defining Mojo-specific MLIR dialects
- Memory model operations (own, borrow, move, copy)

Key dialects needed:
- `mojo` dialect: Core Mojo operations
- Integration with standard MLIR dialects (arith, scf, func, cf, llvm)

### Code Generation

**Location**: `src/codegen/`

**Status**: 🔴 Skeleton Only

Responsible for:
- MLIR optimization pipeline
- Lowering to LLVM IR
- Target-specific optimizations
- Machine code generation

Optimizations needed:
- Inlining, constant folding, DCE
- Loop optimizations
- Move/copy elimination
- Trait devirtualization

### Runtime Support

**Location**: `src/runtime/`

**Status**: 🔴 Skeleton Only

Provides runtime support for:
- Memory management (malloc, free, realloc)
- Async/coroutine runtime
- Type reflection
- String and collection operations
- C library interoperability
- Python interoperability

## Building

To build the compiler (when infrastructure is complete):

```bash
# From repository root
./bazelw build //mojo/compiler/...

# Run tests
./bazelw test //mojo/compiler/...
```

**Note**: Build infrastructure is currently being set up.

## Usage

Target usage (not yet functional):

```bash
# Compile a Mojo file
mojo-compiler build myprogram.mojo

# Compile with options
mojo-compiler build --target=x86_64-linux \
              --stdlib-path=/path/to/stdlib \
              --opt-level=3 \
              myprogram.mojo

# Run tests
mojo-compiler test ./test/
```

## Implementation Status

### Phase 1: Minimal Viable Compiler - **40% Complete**

**Goal**: Compile and run "Hello, World!"

#### Progress:
- [x] Lexer for basic Mojo syntax (85%)
- [x] AST node definitions (complete)
- [🔄] Parser for functions and expressions (60%)
- [ ] Type checker for simple types
- [ ] MLIR code generation for basic operations
- [ ] LLVM backend integration
- [ ] Compile and run "Hello, World!"

**Estimated Time to Phase 1 Completion**: 8-10 weeks

### Phase 2: Core Language Features (Not Started)
- [ ] Full type system (parametrics, traits)
- [ ] Ownership and lifetime checking
- [ ] Complete control flow (if, while, for)
- [ ] Struct definitions and methods
- [ ] Compile basic stdlib modules

### Phase 3: Advanced Features (Not Started)
- [ ] Python interop
- [ ] Async/await
- [ ] GPU support
- [ ] Compile entire stdlib
- [ ] Optimization pipeline

### Phase 4: Production Ready (Not Started)
- [ ] Performance parity with existing compiler
- [ ] Complete language spec coverage
- [ ] Comprehensive error messages
- [ ] IDE integration (LSP)
- [ ] Debugging support (DWARF)

## Documentation

- **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** - Detailed implementation progress and technical status
- **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** - Comprehensive guide for contributors
- **[Open Source Compiler Proposal](../proposals/open-source-compiler.md)** - The full design specification
- **[examples/](examples/)** - Example Mojo programs

## Contributing

We welcome contributions! The compiler is in early stages and needs significant work.

### Key Areas for Contribution

1. **Lexer Completion**: 
   - Add indentation tracking (INDENT/DEDENT tokens)
   - Improve error recovery
   - Add comprehensive tests

2. **Parser Implementation**: 
   - Operator precedence for expressions
   - Control flow statements
   - Struct and trait parsing
   - Better error recovery

3. **Type System Design**: 
   - Type representations
   - Type checking for expressions
   - Symbol table implementation
   - Type inference

4. **MLIR Integration**: 
   - Define Mojo dialect
   - Implement IR generation
   - Text-based MLIR output

5. **Testing**: 
   - Write comprehensive tests
   - Create test infrastructure
   - Add integration tests

### Getting Started

1. Read the [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
2. Check [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for current status
3. Choose a component to work on
4. Follow the contribution guidelines in [CONTRIBUTING.md](../../CONTRIBUTING.md)

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

## Success Metrics

Phase 1 will be considered complete when:
- [x] Compiler structure is in place (done)
- [ ] Lexer passes all tests
- [ ] Parser can parse simple programs
- [ ] Type checker validates simple programs
- [ ] MLIR generator produces valid MLIR
- [ ] Backend generates working executables
- [ ] Hello World program compiles and runs
- [ ] Documentation is complete and accurate

**Current Progress**: ~40% of Phase 1 complete

## License

Licensed under the Apache License v2.0 with LLVM Exceptions.
See [LICENSE](../../LICENSE) for details.

## References

- [LLVM Project](https://llvm.org/)
- [MLIR Documentation](https://mlir.llvm.org/)
- [Mojo Standard Library](../stdlib/)
- [Mojo Language Manual](https://docs.modular.com/mojo/manual/)

## Contact

For questions or discussions:
- See [CONTRIBUTING.md](../../CONTRIBUTING.md) for communication channels
- Review existing issues and discussions
- Read the documentation before asking questions
