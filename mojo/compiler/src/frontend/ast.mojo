# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Abstract Syntax Tree node definitions for the Mojo compiler.

This module defines all AST node types used by the parser.
The AST represents the syntactic structure of Mojo programs.
"""

from collections import List
from .source_location import SourceLocation


@value
struct ASTNodeKind:
    """Represents the kind of an AST node."""
    
    # Top-level constructs
    alias MODULE = 0
    alias FUNCTION = 1
    alias STRUCT = 2
    alias TRAIT = 3
    
    # Statements
    alias VAR_DECL = 10
    alias RETURN_STMT = 11
    alias EXPR_STMT = 12
    alias IF_STMT = 13
    alias WHILE_STMT = 14
    alias FOR_STMT = 15
    alias PASS_STMT = 16
    alias BREAK_STMT = 17
    alias CONTINUE_STMT = 18
    
    # Expressions
    alias BINARY_EXPR = 20
    alias UNARY_EXPR = 21
    alias CALL_EXPR = 22
    alias IDENTIFIER_EXPR = 23
    alias INTEGER_LITERAL = 24
    alias FLOAT_LITERAL = 25
    alias STRING_LITERAL = 26
    alias BOOL_LITERAL = 27
    
    # Types
    alias TYPE_NAME = 30
    alias PARAMETRIC_TYPE = 31
    
    var kind: Int
    
    fn __init__(inout self, kind: Int):
        self.kind = kind


struct ModuleNode:
    """Represents a Mojo module (file).
    
    A module contains top-level declarations like functions, structs, and imports.
    """
    
    var declarations: List[ASTNodeRef]
    var location: SourceLocation
    
    fn __init__(inout self, location: SourceLocation):
        """Initialize a module node.
        
        Args:
            location: Source location of the module.
        """
        self.declarations = List[ASTNodeRef]()
        self.location = location
    
    fn add_declaration(inout self, decl: ASTNodeRef):
        """Add a declaration to the module.
        
        Args:
            decl: The declaration to add.
        """
        self.declarations.append(decl)


struct FunctionNode:
    """Represents a function definition.
    
    Example: fn add(a: Int, b: Int) -> Int: return a + b
    """
    
    var name: String
    var parameters: List[ParameterNode]
    var return_type: TypeNode
    var body: List[ASTNodeRef]
    var location: SourceLocation
    
    fn __init__(
        inout self,
        name: String,
        location: SourceLocation
    ):
        """Initialize a function node.
        
        Args:
            name: The function name.
            location: Source location of the function.
        """
        self.name = name
        self.parameters = List[ParameterNode]()
        self.return_type = TypeNode("None", location)
        self.body = List[ASTNodeRef]()
        self.location = location


struct ParameterNode:
    """Represents a function parameter.
    
    Example: a: Int
    """
    
    var name: String
    var param_type: TypeNode
    var location: SourceLocation
    
    fn __init__(
        inout self,
        name: String,
        param_type: TypeNode,
        location: SourceLocation
    ):
        """Initialize a parameter node.
        
        Args:
            name: The parameter name.
            param_type: The parameter type.
            location: Source location of the parameter.
        """
        self.name = name
        self.param_type = param_type
        self.location = location


struct TypeNode:
    """Represents a type annotation.
    
    Example: Int, String, List[Int]
    """
    
    var name: String
    var location: SourceLocation
    
    fn __init__(inout self, name: String, location: SourceLocation):
        """Initialize a type node.
        
        Args:
            name: The type name.
            location: Source location of the type.
        """
        self.name = name
        self.location = location


struct VarDeclNode:
    """Represents a variable declaration.
    
    Example: var x: Int = 42
    """
    
    var name: String
    var var_type: TypeNode
    var initializer: ASTNodeRef
    var location: SourceLocation
    
    fn __init__(
        inout self,
        name: String,
        var_type: TypeNode,
        initializer: ASTNodeRef,
        location: SourceLocation
    ):
        """Initialize a variable declaration node.
        
        Args:
            name: The variable name.
            var_type: The variable type.
            initializer: The initial value expression.
            location: Source location of the declaration.
        """
        self.name = name
        self.var_type = var_type
        self.initializer = initializer
        self.location = location


struct ReturnStmtNode:
    """Represents a return statement.
    
    Example: return x + y
    """
    
    var value: ASTNodeRef
    var location: SourceLocation
    
    fn __init__(inout self, value: ASTNodeRef, location: SourceLocation):
        """Initialize a return statement node.
        
        Args:
            value: The value to return (may be None).
            location: Source location of the statement.
        """
        self.value = value
        self.location = location


struct BinaryExprNode:
    """Represents a binary expression.
    
    Example: a + b, x * y, a == b
    """
    
    var operator: String
    var left: ASTNodeRef
    var right: ASTNodeRef
    var location: SourceLocation
    
    fn __init__(
        inout self,
        operator: String,
        left: ASTNodeRef,
        right: ASTNodeRef,
        location: SourceLocation
    ):
        """Initialize a binary expression node.
        
        Args:
            operator: The operator symbol (+, -, *, /, ==, etc.).
            left: The left operand.
            right: The right operand.
            location: Source location of the expression.
        """
        self.operator = operator
        self.left = left
        self.right = right
        self.location = location


struct CallExprNode:
    """Represents a function call expression.
    
    Example: print("Hello"), add(1, 2)
    """
    
    var callee: String
    var arguments: List[ASTNodeRef]
    var location: SourceLocation
    
    fn __init__(
        inout self,
        callee: String,
        location: SourceLocation
    ):
        """Initialize a call expression node.
        
        Args:
            callee: The function name being called.
            location: Source location of the call.
        """
        self.callee = callee
        self.arguments = List[ASTNodeRef]()
        self.location = location
    
    fn add_argument(inout self, arg: ASTNodeRef):
        """Add an argument to the call.
        
        Args:
            arg: The argument expression.
        """
        self.arguments.append(arg)


struct IdentifierExprNode:
    """Represents an identifier expression.
    
    Example: x, variable_name
    """
    
    var name: String
    var location: SourceLocation
    
    fn __init__(inout self, name: String, location: SourceLocation):
        """Initialize an identifier expression node.
        
        Args:
            name: The identifier name.
            location: Source location of the identifier.
        """
        self.name = name
        self.location = location


struct IntegerLiteralNode:
    """Represents an integer literal.
    
    Example: 42, 0, -10
    """
    
    var value: String
    var location: SourceLocation
    
    fn __init__(inout self, value: String, location: SourceLocation):
        """Initialize an integer literal node.
        
        Args:
            value: The integer value as a string.
            location: Source location of the literal.
        """
        self.value = value
        self.location = location


struct FloatLiteralNode:
    """Represents a float literal.
    
    Example: 3.14, 0.5, -2.718
    """
    
    var value: String
    var location: SourceLocation
    
    fn __init__(inout self, value: String, location: SourceLocation):
        """Initialize a float literal node.
        
        Args:
            value: The float value as a string.
            location: Source location of the literal.
        """
        self.value = value
        self.location = location


struct StringLiteralNode:
    """Represents a string literal.
    
    Example: "Hello, World!", 'test'
    """
    
    var value: String
    var location: SourceLocation
    
    fn __init__(inout self, value: String, location: SourceLocation):
        """Initialize a string literal node.
        
        Args:
            value: The string content.
            location: Source location of the literal.
        """
        self.value = value
        self.location = location


# Type alias for AST node references
# In a real implementation, this would be a variant/union type or trait object
alias ASTNodeRef = Int  # Placeholder - would be a proper reference type
