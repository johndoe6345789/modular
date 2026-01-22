# Example: Hello World

This is a simple example to demonstrate the compiler structure.

```mojo
fn main():
    print("Hello, World!")
```

## Compilation Steps

1. **Lexer Output** (tokens):
```
FN, IDENTIFIER("main"), LEFT_PAREN, RIGHT_PAREN, COLON, NEWLINE,
INDENT, IDENTIFIER("print"), LEFT_PAREN, STRING_LITERAL("Hello, World!"), 
RIGHT_PAREN, NEWLINE, DEDENT, EOF
```

2. **Parser Output** (AST):
```
Module
  Function "main"
    Parameters: []
    Return Type: None
    Body:
      CallExpr
        Callee: Identifier "print"
        Arguments: [StringLiteral "Hello, World!"]
```

3. **Type Checker Output**:
- Verify `print` is a builtin function
- Check argument types match `print` signature
- Infer return type of `main` is `None`

4. **MLIR Output**:
```mlir
module {
  func.func @main() {
    %0 = arith.constant "Hello, World!" : !llvm.ptr
    call @print(%0) : (!llvm.ptr) -> ()
    return
  }
}
```

5. **LLVM IR Output**:
```llvm
define void @main() {
entry:
  %0 = getelementptr inbounds [14 x i8], [14 x i8]* @str_0, i32 0, i32 0
  call void @print(i8* %0)
  ret void
}

@str_0 = private unnamed_addr constant [14 x i8] c"Hello, World!\00"
```

6. **Native Code**: Compiled to x86_64/ARM64 machine code
