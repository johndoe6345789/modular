# Open Source Mojo Compiler Design Specification

**Status**: Proposed  
**Author**: Community  
**Date**: 2026-01-22

## Executive Summary

This document outlines the design for an open source Mojo compiler implementation that can compile Mojo source code and work with the existing open source standard library. The goal is to enable the community to fork and create an independent, fully open source Mojo toolchain.

## Background

Currently, the Mojo ecosystem consists of:
- **Open Source**: Standard library (`/mojo/stdlib/`), examples, documentation
- **Closed Source**: Mojo compiler, compiler runtime, MLIR dialects (`pop`, `kgen`, `lit`)

This creates a barrier for community participation in compiler development and limits the ability to create truly independent implementations. An open source compiler would enable:
- Full transparency in language implementation
- Community-driven compiler improvements
- Independent ports to new platforms
- Educational use cases
- Fork-ability for specialized use cases

## Goals

1. **Full Language Support**: Implement complete Mojo language specification
2. **Standard Library Compatibility**: Work seamlessly with the existing open source stdlib
3. **Performance**: Achieve competitive performance with existing implementation—maintaining Mojo's speed advantage over Python through zero-cost abstractions, LLVM optimizations, and efficient low-level code generation
4. **C Library Interoperability**: Preserve seamless integration with C libraries and system APIs with zero overhead
5. **Modularity**: Clean separation of concerns for maintainability
6. **Extensibility**: Easy to add new backends and optimizations
7. **LLVM Foundation**: Build on proven LLVM infrastructure

## Non-Goals

- Binary compatibility with existing Modular-distributed Mojo compiler
- Support for proprietary Modular Platform features
- Backward compatibility with internal/private APIs

## Architecture Overview

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
│                    Mojo AST/IR                               │
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
│  • Inlining, constant folding, DCE, etc.                    │
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

## Component Specifications

### 1. Frontend: Lexer and Parser

**Responsibilities**:
- Tokenize Mojo source code
- Build Abstract Syntax Tree (AST)
- Report syntax errors with helpful diagnostics

**Key Features**:
- Support for all Mojo syntax (struct, fn, var, def, etc.)
- Parameter blocks `[T: Type]`
- Decorators (`@value`, `@register_passable`, etc.)
- Python interop syntax

**Implementation Approach**:
- Hand-written recursive descent parser or parser generator (ANTLR/Bison)
- UTF-8 source encoding support
- Location tracking for error messages

**Interfaces**:
```mojo
struct Token:
    var kind: TokenKind
    var text: String
    var location: SourceLocation

struct AST:
    var root: ASTNode
    
trait ASTNode:
    fn accept[V: ASTVisitor](self, visitor: V)
```

### 2. Semantic Analysis

**Responsibilities**:
- Type checking and inference
- Name resolution and scoping
- Trait resolution
- Lifetime and ownership analysis
- Compile-time evaluation

**Key Features**:
- Parametric type system
- Trait-based generics
- Value semantics and ownership checking
- Reference lifetime validation

**Challenges**:
- Compile-time parameter evaluation
- Type parameter inference
- Lifetime provenance tracking

### 3. MLIR Integration

**Core Dialects Required**:

#### 3.1 Mojo-Specific Dialects

Create open source equivalents for:

**`mojo` dialect** (replaces private `pop`, `kgen` dialects):
- Basic Mojo operations
- Value semantics operations
- Parametric types representation
- Trait dispatch

Example operations:
```mlir
// Function definition
mojo.func @example(%arg0: !mojo.value<Int>) -> !mojo.value<Int>

// Struct type
!mojo.struct<"MyStruct", fields: [i64, f64]>

// Parametric call
mojo.parametric_call @function[%T = i64](%arg0) : (!mojo.value<i64>) -> !mojo.value<i64>

// Trait dispatch
mojo.trait_call @Hashable::hash(%arg0) : (!mojo.value<String>) -> !mojo.value<Int>
```

#### 3.2 Standard LLVM Dialects

Leverage existing MLIR infrastructure:
- `arith`: Arithmetic operations
- `scf`: Structured control flow
- `func`: Function abstraction
- `cf`: Control flow
- `llvm`: LLVM IR dialect
- `gpu`: GPU operations (CUDA, ROCm)

#### 3.3 Memory Model

```mlir
// Owned value
%owned = mojo.own %value : !mojo.value<String>

// Borrowed reference
%borrowed = mojo.borrow %value : !mojo.value<String>

// Mutable reference
%mut = mojo.mut_borrow %value : !mojo.value<String>

// Move operation
%moved = mojo.move %value : !mojo.value<String>

// Copy operation
%copied = mojo.copy %value : !mojo.value<String>
```

### 4. Compiler Runtime

Replace the closed-source compiler runtime with an open implementation.

**Required Components**:

#### 4.1 Memory Management
```mojo
# Allocation
fn malloc(size: Int) -> UnsafePointer[Byte]
fn free(ptr: UnsafePointer[Byte])
fn realloc(ptr: UnsafePointer[Byte], new_size: Int) -> UnsafePointer[Byte]

# Reference counting (if used)
fn retain[T: AnyType](ptr: UnsafePointer[T])
fn release[T: AnyType](ptr: UnsafePointer[T])
```

#### 4.2 Async Runtime
```mojo
# Coroutines and async support
fn create_coroutine() -> CoroutineHandle
fn suspend_coroutine(handle: CoroutineHandle)
fn resume_coroutine(handle: CoroutineHandle)
fn destroy_coroutine(handle: CoroutineHandle)

# Async executor
struct AsyncExecutor:
    fn spawn[F: AsyncFn](task: F)
    fn run_until_complete[F: AsyncFn](task: F)
```

#### 4.3 Type Reflection
```mojo
# Runtime type information
fn get_type_info[T: AnyType]() -> TypeInfo
fn type_name[T: AnyType]() -> String
```

#### 4.4 String and Collection Support
```mojo
# String operations that need runtime support
fn string_concat(a: String, b: String) -> String
fn string_format(template: String, args: VariadicList) -> String

# Dynamic collections
fn list_append[T: AnyType](list: List[T], item: T)
fn dict_insert[K: KeyType, V: AnyType](dict: Dict[K,V], key: K, value: V)
```

### 5. Standard Library Integration

**Interface Requirements**:

The compiler must provide these built-in operations that the stdlib depends on:

#### 5.1 Builtin Types
- `Int`, `UInt`, `Float64`, `Bool`
- `String`, `StringLiteral`
- `SIMD[type, size]`
- `Pointer[T]`, `Reference[T]`

#### 5.2 Builtin Functions
```mojo
fn print(*args: *T)
fn len[T: Sized](value: T) -> Int
fn sizeof[T: AnyType]() -> Int
fn alignof[T: AnyType]() -> Int
fn rebind[T: AnyType](value: AnyType) -> T
```

#### 5.3 Magic Methods
- `__init__`, `__del__`
- `__copyinit__`, `__moveinit__`
- `__getattr__`, `__setattr__`
- `__getitem__`, `__setitem__`
- `__add__`, `__sub__`, `__mul__`, etc.

### 6. Build System Integration

**Requirements**:
- Build stdlib from source
- Handle dependencies between modules
- Support incremental compilation
- Package management integration

**Proposed Structure**:
```bash
# Compiler invocation
mojo-oss build --target=x86_64-linux \
              --stdlib-path=/path/to/stdlib \
              --opt-level=3 \
              myprogram.mojo

# Package building
mojo-oss package build --manifest=mojo.toml

# Testing
mojo-oss test ./test/
```

**Build Configuration** (`mojo.toml`):
```toml
[package]
name = "myapp"
version = "0.1.0"
edition = "2026"

[dependencies]
stdlib = { path = "../stdlib" }

[build]
target = "x86_64-linux"
opt-level = 3
```

### 7. C Library Interoperability

**Requirements**:
- Call C functions from Mojo code
- Use C types and structs
- Link with C libraries (static and dynamic)
- Zero-cost abstractions over C APIs
- ABI compatibility with C

**Implementation**:
- Foreign Function Interface (FFI) layer
- C type mapping to Mojo types
- External function declarations
- Proper calling convention support (cdecl, stdcall, etc.)
- Header file processing (C interop declarations)

**Example Usage**:
```mojo
from sys.ffi import external_call

# Declare C function
fn c_strlen(s: UnsafePointer[UInt8]) -> Int:
    return external_call["strlen", Int](s)

# Use C library (e.g., libc math)
@external
fn sin(x: Float64) -> Float64

fn use_c_math():
    let result = sin(3.14159 / 2)  # Calls C's sin function
    print("sin(π/2) =", result)

# Work with C structs
@value
@register_passable("trivial")
struct CTimeSpec:
    var tv_sec: Int64
    var tv_nsec: Int64

@external
fn clock_gettime(clk_id: Int, tp: UnsafePointer[CTimeSpec]) -> Int
```

**Design Principles**:
- **Zero-cost abstraction**: C calls should have no overhead compared to native C
- **Type safety**: Provide safe wrappers while allowing unsafe access when needed
- **Compatibility**: Work with existing C libraries without modification
- **Performance**: Direct function calls, no runtime overhead
- **ABI stability**: Maintain stable C calling conventions

This preserves Mojo's strength in systems programming and allows seamless integration with the vast ecosystem of C libraries (database drivers, system APIs, scientific computing libraries, etc.).

### 8. Python Interoperability

**Requirements**:
- Import Python modules
- Call Python functions from Mojo
- Pass data between Mojo and Python
- Use Python objects in Mojo code

**Implementation**:
- Embed Python interpreter (CPython)
- FFI layer for object conversion
- Python C API wrapper

```mojo
from python import Python

fn use_numpy():
    let np = Python.import_module("numpy")
    let arr = np.array([1, 2, 3])
    print(arr.shape)
```

### 9. Optimization Pipeline

**High-Level Optimizations**:
- Inlining
- Constant propagation and folding
- Dead code elimination
- Loop optimizations (unrolling, vectorization)
- Common subexpression elimination

**Mojo-Specific Optimizations**:
- Value lifetime optimization
- Move elimination
- Copy elimination (when safe)
- Parametric specialization
- Trait devirtualization

**LLVM Backend**:
- Leverage LLVM optimization passes
- Target-specific code generation
- Auto-vectorization
- Link-time optimization (LTO)

### 10. GPU Support

**Requirements**:
- CUDA backend for NVIDIA GPUs
- ROCm backend for AMD GPUs
- Metal backend for Apple Silicon
- Generic GPU abstraction layer

**Approach**:
- Use MLIR GPU dialect
- Lower to target-specific IRs (PTX, AMDGPU, Metal)
- Support for GPU kernels in Mojo syntax

```mojo
from gpu import *

@kernel
fn vector_add[size: Int](
    a: DeviceBuffer[Float32, size],
    b: DeviceBuffer[Float32, size],
    c: DeviceBuffer[Float32, size]
):
    let idx = thread_id()
    if idx < size:
        c[idx] = a[idx] + b[idx]
```

## Implementation Phases

### Phase 1: Minimal Viable Compiler (3-6 months)
- [ ] Lexer and parser for basic Mojo syntax
- [ ] Type checker for simple types
- [ ] MLIR code generation for basic operations
- [ ] LLVM backend integration
- [ ] Compile and run "Hello, World!"

### Phase 2: Core Language Features (6-12 months)
- [ ] Full type system (parametrics, traits)
- [ ] Ownership and lifetime checking
- [ ] Complete control flow (if, while, for)
- [ ] Struct definitions and methods
- [ ] Compile basic stdlib modules

### Phase 3: Advanced Features (12-18 months)
- [ ] Python interop
- [ ] Async/await
- [ ] GPU support
- [ ] Compile entire stdlib
- [ ] Optimization pipeline

### Phase 4: Production Ready (18-24 months)
- [ ] Performance parity with existing compiler
- [ ] Complete language spec coverage
- [ ] Comprehensive error messages
- [ ] IDE integration (LSP)
- [ ] Debugging support (DWARF)

## Technical Challenges

### 1. Parametric Type System
**Challenge**: Compile-time evaluation of type parameters  
**Solution**: Implement constexpr evaluation engine, lazy type instantiation

### 2. Ownership and Lifetimes
**Challenge**: Proving safety without explicit annotations  
**Solution**: Borrow checker similar to Rust, lifetime inference

### 3. MLIR Dialect Design
**Challenge**: Designing dialects that efficiently represent Mojo semantics  
**Solution**: Iterative design with community feedback, study existing implementations

### 4. Standard Library ABI
**Challenge**: Maintaining compatibility with existing stdlib  
**Solution**: Document ABI requirements, automated ABI testing

### 5. Performance
**Challenge**: Matching performance of highly optimized existing compiler  
**Solution**: Profile-guided optimization, leverage LLVM, community benchmarking

## Community and Governance

### Project Structure
- **Core Team**: 3-5 maintainers responsible for design decisions
- **Contributors**: Open contribution model following existing CONTRIBUTING.md
- **Special Interest Groups**: GPU, async, optimization, etc.

### Communication
- GitHub issues for bug tracking
- RFC process for major changes (follow `/mojo/proposals/` pattern)
- Monthly community meetings
- Discord/Forum for real-time discussion

### Testing and Quality
- Comprehensive test suite
- Continuous integration
- Fuzzing for parser and type checker
- Performance benchmarking against existing compiler

## Success Metrics

1. **Correctness**: Pass 100% of stdlib test suite
2. **Performance**: Within 20% of existing compiler on benchmarks
3. **Completeness**: Support all documented Mojo language features
4. **Adoption**: Used by at least 5 significant projects
5. **Sustainability**: 10+ regular contributors

## Alternative Approaches

### Option A: Fork LLVM Flang/Clang
**Pros**: Mature infrastructure, battle-tested  
**Cons**: Complex codebase, C++-heavy, different language model

### Option B: Build on Rust Compiler Infrastructure
**Pros**: Modern design, similar ownership model  
**Cons**: Different ecosystem, fewer MLIR integrations

### Option C: Pure Python Implementation
**Pros**: Easy prototyping, Python interop  
**Cons**: Performance concerns, complex optimization

**Recommendation**: Build custom compiler with MLIR/LLVM (this proposal)

## Resources Required

### Technical Infrastructure
- CI/CD: GitHub Actions + self-hosted runners for GPU testing
- Hardware: Access to various GPU architectures (NVIDIA, AMD, Apple)
- Storage: Binary artifacts, test results, benchmarks

### Documentation
- Compiler architecture guide
- MLIR dialect specifications
- Contribution guide for compiler development
- User documentation

### Community
- Core maintainers with LLVM/compiler experience
- GPU programming experts
- Type theory experts for type system design

## Timeline and Milestones

**Year 1: Foundation**
- Q1: Design finalization, community formation
- Q2: Parser and basic type checking
- Q3: MLIR integration, basic code generation
- Q4: Hello World compilation

**Year 2: Core Implementation**
- Q1: Complete type system implementation
- Q2: Ownership and lifetime checking
- Q3: Stdlib compatibility layer
- Q4: Basic optimization pipeline

**Year 3: Production Readiness**
- Q1: Python interop and GPU support
- Q2: Performance optimization
- Q3: Tooling (debugger, profiler, LSP)
- Q4: 1.0 release candidate

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Insufficient contributors | High | Early community building, good documentation |
| Compatibility issues with stdlib | High | Regular testing, ABI documentation |
| Performance gap too large | Medium | Focus on optimization early, leverage LLVM |
| MLIR dialect design flaws | High | Early prototyping, expert consultation |
| Legal/licensing issues | Medium | Legal review, clean-room implementation |

## Open Questions

1. Should we maintain source compatibility or allow improvements?
2. How to handle closed-source features (if users depend on them)?
3. What's the governance model for accepting RFCs?
4. Should we support older stdlib versions?
5. Binary distribution strategy?

## Conclusion

An open source Mojo compiler is technically feasible and would provide significant value to the community. By building on MLIR and LLVM infrastructure, we can create a production-quality compiler within 2-3 years with sufficient community support.

The key to success is:
1. Clear technical specification (this document)
2. Strong community governance
3. Incremental development with frequent milestones
4. Focus on stdlib compatibility
5. Performance as a first-class concern

## References

- [LLVM Project](https://llvm.org/)
- [MLIR Documentation](https://mlir.llvm.org/)
- [Mojo Standard Library](https://github.com/modular/modular/tree/main/mojo/stdlib)
- [Mojo Language Manual](https://docs.modular.com/mojo/manual/)
- [Rust Compiler Architecture](https://rustc-dev-guide.rust-lang.org/)

## Appendix A: Example MLIR Lowering

**Mojo Source**:
```mojo
fn add(a: Int, b: Int) -> Int:
    return a + b
```

**MLIR Representation**:
```mlir
func.func @add(%arg0: i64, %arg1: i64) -> i64 {
  %result = arith.addi %arg0, %arg1 : i64
  return %result : i64
}
```

## Appendix B: Compiler API

```mojo
from compiler import *

struct CompilerOptions:
    var target: String
    var opt_level: Int
    var stdlib_path: String
    var debug: Bool

fn compile(source_file: String, options: CompilerOptions) raises -> Executable:
    let ast = parse(source_file)
    let typed_ast = type_check(ast)
    let mlir = generate_mlir(typed_ast)
    let optimized = optimize(mlir, options.opt_level)
    let binary = codegen(optimized, options.target)
    return binary
```

## Appendix C: Required MLIR Operations

**Core Mojo Operations**:
- `mojo.struct_def`: Define a struct type
- `mojo.trait_def`: Define a trait
- `mojo.fn_def`: Define a function
- `mojo.param_value`: Compile-time parameter
- `mojo.trait_impl`: Implement trait for type
- `mojo.call_trait`: Dynamic dispatch through trait
- `mojo.copy`: Explicit copy operation
- `mojo.move`: Move operation
- `mojo.borrow`: Create borrowed reference
- `mojo.own`: Take ownership

## Appendix D: Contributing Guide

For those interested in contributing to the open source compiler:

1. **Start Small**: Fix bugs, improve error messages
2. **Read the Code**: Study MLIR and LLVM internals
3. **Write Tests**: Add test cases for language features
4. **Documentation**: Help document the architecture
5. **Proposals**: Submit RFCs for new features

See the main [CONTRIBUTING.md](../../CONTRIBUTING.md) for general guidelines.
