# Mojo Runtime Library

This directory contains the runtime library for the Mojo compiler. The runtime provides essential functions that are called by compiled Mojo code.

## Overview

The Mojo runtime library is implemented in C for portability and ease of integration with system linkers. It provides:

- **Print functions**: Support for printing different data types to stdout
- **Memory management**: Basic allocation/deallocation (future)
- **Exception handling**: Runtime support for exceptions (future)

## Building

To build the runtime library:

```bash
cd runtime
make
```

This will create `libmojo_runtime.a`, a static library that is linked with compiled Mojo programs.

To clean build artifacts:

```bash
make clean
```

## API Reference

### Print Functions

All print functions output to stdout followed by a newline.

#### `_mojo_print_string(const char* str)`
Print a null-terminated string.

**Parameters:**
- `str`: Pointer to null-terminated string

**Example:**
```llvm
declare void @_mojo_print_string(i8*)

@.str = private constant [14 x i8] c"Hello, World!\00"

define i32 @main() {
  %0 = getelementptr [14 x i8], [14 x i8]* @.str, i32 0, i32 0
  call void @_mojo_print_string(i8* %0)
  ret i32 0
}
```

#### `_mojo_print_int(int64_t value)`
Print a 64-bit signed integer.

**Parameters:**
- `value`: 64-bit signed integer value

**Example:**
```llvm
declare void @_mojo_print_int(i64)

define i32 @main() {
  call void @_mojo_print_int(i64 42)
  ret i32 0
}
```

#### `_mojo_print_float(double value)`
Print a 64-bit floating point number.

**Parameters:**
- `value`: Double precision floating point value

**Example:**
```llvm
declare void @_mojo_print_float(double)

define i32 @main() {
  call void @_mojo_print_float(double 3.14159)
  ret i32 0
}
```

#### `_mojo_print_bool(bool value)`
Print a boolean value as "True" or "False".

**Parameters:**
- `value`: Boolean value (0 = false, non-zero = true)

**Example:**
```llvm
declare void @_mojo_print_bool(i1)

define i32 @main() {
  call void @_mojo_print_bool(i1 true)
  ret i32 0
}
```

## Integration with Compiler

The compiler backend automatically:

1. Generates declarations for runtime functions in LLVM IR
2. Links compiled object files with `libmojo_runtime.a`
3. Uses the system linker to create the final executable

The runtime library must be built before attempting to compile Mojo programs.

## Requirements

- C compiler (gcc, clang, or compatible)
- Standard C library
- `ar` archiver for creating static libraries

## File Structure

```
runtime/
├── print.c           # Print function implementations
├── Makefile          # Build system
└── README.md         # This file
```

## Future Enhancements

Planned additions to the runtime:

- Memory management functions (malloc/free wrappers)
- Exception handling support
- String manipulation utilities
- I/O functions
- Math library functions
- Concurrency primitives

## License

Copyright (c) 2025, Modular Inc. All rights reserved.

Licensed under the Apache License v2.0 with LLVM Exceptions.
See LICENSE file for details.
