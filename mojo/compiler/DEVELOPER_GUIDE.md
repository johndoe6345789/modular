# Developer Guide for the Open Source Mojo Compiler

This guide is for developers contributing to the Mojo compiler implementation.

## Architecture Overview

The compiler follows a traditional multi-pass architecture:

```
Source Code
    ↓
┌─────────────────┐
│  Lexer          │  Tokenization
│  (lexer.mojo)   │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Parser         │  AST Construction
│  (parser.mojo)  │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Type Checker   │  Semantic Analysis
│  (type_checker) │
└────────┬────────┘
         ↓
┌─────────────────┐
│  MLIR Generator │  IR Generation
│  (mlir_gen)     │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Optimizer      │  Optimization
│  (optimizer)    │
└────────┬────────┘
         ↓
┌─────────────────┐
│  LLVM Backend   │  Code Generation
│  (llvm_backend) │
└────────┬────────┘
         ↓
    Executable
```

## Component Details

### 1. Lexer (`src/frontend/lexer.mojo`)

**Purpose**: Convert source text into a stream of tokens.

**Key Types**:
- `TokenKind`: Enumeration of all token types
- `Token`: A lexical token with kind, text, and location
- `Lexer`: The tokenizer

**Implementation Status**: ✅ Core functionality complete

**Key Methods**:
- `next_token()`: Get the next token from the source
- `peek_char()`: Look at current character without consuming
- `advance()`: Move to next character
- `skip_whitespace()`: Skip spaces and tabs
- `skip_comment()`: Skip # comments
- `read_identifier()`: Parse identifiers and keywords
- `read_number()`: Parse integer and float literals
- `read_string()`: Parse string literals with escapes

**Usage Example**:
```mojo
var lexer = Lexer("fn main(): print('Hello')", "example.mojo")
var token = lexer.next_token()
while token.kind.kind != TokenKind.EOF:
    print(token.text)
    token = lexer.next_token()
```

**Future Work**:
- Add indentation tracking (INDENT/DEDENT tokens)
- Better error recovery
- Unicode support
- Performance optimizations

### 2. Parser (`src/frontend/parser.mojo`, `src/frontend/ast.mojo`)

**Purpose**: Build an Abstract Syntax Tree from tokens.

**Key Types**:
- `Parser`: The parser
- `AST`: The abstract syntax tree
- `ModuleNode`: Represents a Mojo file
- `FunctionNode`: Function definition
- `ExpressionNodes`: Various expression types
- `StatementNodes`: Various statement types

**Implementation Status**: 🔄 Partial implementation

**Key Methods**:
- `parse()`: Parse entire module
- `parse_module()`: Parse top-level declarations
- `parse_function()`: Parse function definitions
- `parse_expression()`: Parse expressions
- `parse_statement()`: Parse statements
- `parse_type()`: Parse type annotations

**Usage Example**:
```mojo
var parser = Parser("fn main(): print('Hello')", "example.mojo")
let ast = parser.parse()
if parser.has_errors():
    for error in parser.errors:
        print(error)
```

**Current Limitations**:
- AST nodes use placeholder references (needs proper node storage)
- No operator precedence parsing yet
- Limited statement types
- No control flow parsing

**Future Work**:
- Implement full expression parsing with precedence climbing
- Add all statement types (if, while, for, etc.)
- Implement struct and trait parsing
- Add decorator support
- Better error recovery

### 3. AST Nodes (`src/frontend/ast.mojo`)

**Purpose**: Define all Abstract Syntax Tree node types.

**Key Node Types**:

#### Top-Level Nodes
- `ModuleNode`: A Mojo source file
- `FunctionNode`: Function definition
- `StructNode`: Struct definition (TODO)
- `TraitNode`: Trait definition (TODO)

#### Statement Nodes
- `VarDeclNode`: Variable declaration (`var x: Int = 5`)
- `ReturnStmtNode`: Return statement
- `IfStmtNode`: If statement (TODO)
- `WhileStmtNode`: While loop (TODO)
- `ForStmtNode`: For loop (TODO)

#### Expression Nodes
- `BinaryExprNode`: Binary operations (`a + b`)
- `UnaryExprNode`: Unary operations (`-x`) (TODO)
- `CallExprNode`: Function calls (`print("hi")`)
- `IdentifierExprNode`: Variable references (`x`)
- `IntegerLiteralNode`: Integer literals (`42`)
- `FloatLiteralNode`: Float literals (`3.14`)
- `StringLiteralNode`: String literals (`"hello"`)

#### Type Nodes
- `TypeNode`: Type annotations
- `ParametricTypeNode`: Generic types (TODO)

**Design Pattern**: Each node contains:
- `location`: Source location for error reporting
- Type-specific fields (name, value, children, etc.)

**Future Work**:
- Implement missing node types
- Add proper node storage/reference mechanism
- Add visitor pattern for tree traversal
- Add pretty-printing for debugging

### 4. Type Checker (`src/semantic/type_checker.mojo`)

**Purpose**: Validate types and semantics.

**Implementation Status**: 🔴 Skeleton only

**Key Components**:
- `TypeChecker`: Main type checking engine
- `SymbolTable`: Name resolution
- `TypeContext`: Type information storage

**Future Work**:
- Implement type representation
- Implement type checking for expressions
- Implement type inference
- Add scope management
- Add error reporting

### 5. MLIR Generator (`src/ir/mlir_gen.mojo`)

**Purpose**: Lower AST to MLIR intermediate representation.

**Implementation Status**: 🔴 Skeleton only

**Approach**:
Two implementation options:

**Option A: Text Generation (Simple)**
- Generate MLIR as text
- Use `mlir-opt` command-line tool for validation
- Use `mlir-translate` to lower to LLVM IR

**Option B: MLIR C++ API (Complex)**
- Link against MLIR libraries via FFI
- Build IR programmatically
- More control and better error checking

**Recommended for Phase 1**: Option A (text generation)

**Future Work**:
- Define Mojo dialect operations
- Implement lowering for each AST node type
- Add MLIR validation
- Implement optimization pipeline

### 6. LLVM Backend (`src/codegen/llvm_backend.mojo`)

**Purpose**: Generate native code from MLIR.

**Implementation Status**: 🔴 Skeleton only

**Approach**:
For Phase 1, use command-line tools:
1. MLIR → LLVM IR: `mlir-translate`
2. LLVM IR → Object file: `llc`
3. Object file → Executable: `clang` or `ld`

**Future Work**:
- Implement MLIR to LLVM IR lowering
- Add target-specific code generation
- Implement linking
- Add optimization passes

## Development Workflow

### Setting Up Development Environment

1. Clone the repository
2. Ensure Mojo nightly is installed
3. Familiarize yourself with the proposal: `mojo/proposals/open-source-compiler.md`

### Making Changes

1. **Choose a component** to work on (see IMPLEMENTATION_STATUS.md)
2. **Understand the interface**: Read the existing code and comments
3. **Implement the feature**: Follow existing code style
4. **Test your changes**: Create test cases
5. **Document your work**: Update comments and documentation

### Code Style

Follow Mojo conventions:
- Use descriptive variable names
- Add docstrings to all public functions
- Use type annotations
- Keep functions focused and small
- Add comments for complex logic

Example:
```mojo
fn parse_expression(inout self) -> ASTNodeRef:
    """Parse an expression with proper operator precedence.
    
    Returns:
        The expression AST node reference.
    """
    # Implementation here
    ...
```

### Testing Strategy

For each component, create tests:

1. **Unit tests**: Test individual functions
2. **Integration tests**: Test component interaction
3. **End-to-end tests**: Test full compilation

Example test structure:
```mojo
fn test_lexer_keywords():
    var lexer = Lexer("fn struct var", "test.mojo")
    
    let tok1 = lexer.next_token()
    assert tok1.kind.kind == TokenKind.FN
    
    let tok2 = lexer.next_token()
    assert tok2.kind.kind == TokenKind.STRUCT
    
    let tok3 = lexer.next_token()
    assert tok3.kind.kind == TokenKind.VAR
```

## Common Patterns

### Error Handling

Use the `errors` list to accumulate errors:
```mojo
fn parse_function(inout self) -> FunctionNode:
    if self.current_token.kind.kind != TokenKind.FN:
        self.error("Expected 'fn' keyword")
        return FunctionNode("error", self.current_token.location)
    # Continue parsing...
```

### Source Locations

Always track source locations for error reporting:
```mojo
let start_location = self.current_token.location
# Parse something...
let node = FunctionNode(name, start_location)
```

### Token Consumption

Use `expect()` for required tokens:
```mojo
if not self.expect(TokenKind(TokenKind.LEFT_PAREN)):
    self.error("Expected '('")
    return
```

## Debugging Tips

1. **Print tokens**: Add debug printing in the lexer to see token stream
2. **Print AST**: Implement a tree printer to visualize the AST
3. **Check locations**: Verify source locations are correct
4. **Test incrementally**: Test each piece in isolation first

## Performance Considerations

Current priorities:
1. **Correctness**: Get it working correctly first
2. **Completeness**: Support full language features
3. **Performance**: Optimize later

Future optimizations:
- String interning for identifiers
- Memory pooling for AST nodes
- Parallel parsing for modules
- Incremental compilation

## Resources

### External References
- [LLVM Documentation](https://llvm.org/docs/)
- [MLIR Documentation](https://mlir.llvm.org/)
- [Compiler Design Book](https://en.wikipedia.org/wiki/Compilers:_Principles,_Techniques,_and_Tools)

### Internal References
- [Compiler Proposal](../proposals/open-source-compiler.md)
- [Implementation Status](IMPLEMENTATION_STATUS.md)
- [Architecture README](README.md)
- [Mojo Standard Library](../stdlib/)

## Getting Help

If you're stuck:
1. Check existing code for similar patterns
2. Read the proposal document
3. Look at MLIR/LLVM documentation
4. Ask questions in discussions

## Contributing Checklist

Before submitting a PR:
- [ ] Code compiles without errors
- [ ] Tests pass (when test infrastructure exists)
- [ ] Documentation is updated
- [ ] Code follows style guidelines
- [ ] Comments explain complex logic
- [ ] IMPLEMENTATION_STATUS.md is updated

## Next Steps for Contributors

High-priority tasks:
1. Complete lexer implementation (indentation tracking)
2. Implement remaining parser functionality
3. Design and implement type system
4. Create test infrastructure
5. Implement MLIR generation (text-based approach)

See IMPLEMENTATION_STATUS.md for detailed task list.
