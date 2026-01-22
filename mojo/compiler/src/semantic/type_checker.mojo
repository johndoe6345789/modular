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

"""Type checker for the Mojo compiler.

The type checker performs semantic analysis on the AST:
- Type checking and type inference
- Name resolution
- Ownership and lifetime checking
- Trait resolution
"""

from collections import List
from ..frontend.parser import AST, Parser
from ..frontend.ast import (
    ModuleNode,
    FunctionNode,
    ASTNodeRef,
    ASTNodeKind,
)
from ..frontend.source_location import SourceLocation
from .symbol_table import SymbolTable
from .type_system import Type, TypeContext


struct TypeChecker:
    """Performs type checking and semantic analysis on an AST.
    
    Responsibilities:
    - Type checking expressions and statements
    - Type inference where types are not explicit
    - Name resolution using symbol tables
    - Ownership and lifetime validation
    - Trait conformance checking
    """
    
    var symbol_table: SymbolTable
    var type_context: TypeContext
    var errors: List[String]
    var parser: Parser  # Reference to parser for node access
    var current_function_return_type: Type  # Track expected return type
    
    fn __init__(inout self, parser: Parser):
        """Initialize the type checker.
        
        Args:
            parser: The parser containing the AST nodes.
        """
        self.symbol_table = SymbolTable()
        self.type_context = TypeContext()
        self.errors = List[String]()
        self.parser = parser
        self.current_function_return_type = Type("Unknown")
    
    fn check(inout self, ast: AST) -> Bool:
        """Type check an entire AST.
        
        Args:
            ast: The AST to type check.
            
        Returns:
            True if type checking succeeded, False if errors were found.
        """
        # Register builtin functions in symbol table
        self._register_builtins()
        
        # Check all declarations in the module
        for i in range(len(ast.root.declarations)):
            let decl_ref = ast.root.declarations[i]
            self.check_node(decl_ref)
        
        return len(self.errors) == 0
    
    fn _register_builtins(inout self):
        """Register builtin functions like print."""
        # print() function accepts any type and returns None
        self.symbol_table.insert("print", Type("Function"))
    
    fn check_node(inout self, node_ref: ASTNodeRef):
        """Type check a single AST node by dispatching based on node kind.
        
        Args:
            node_ref: The node reference to type check.
        """
        # Get node kind to dispatch appropriately
        let kind = self.parser.node_store.get_node_kind(node_ref)
        
        if kind == ASTNodeKind.FUNCTION:
            _ = self.check_function(node_ref)
        elif kind == ASTNodeKind.VAR_DECL:
            self.check_statement(node_ref)
        elif kind == ASTNodeKind.RETURN_STMT:
            self.check_statement(node_ref)
        elif self.parser.node_store.is_expression(node_ref):
            _ = self.check_expression(node_ref)
        elif self.parser.node_store.is_statement(node_ref):
            self.check_statement(node_ref)
    
    fn check_function(inout self, node_ref: ASTNodeRef) -> Type:
        """Type check a function definition.
        
        Args:
            node_ref: The function node reference (index into parser.function_nodes).
            
        Returns:
            The function type.
        """
        # Function nodes are stored separately - for Phase 1, we'll access via parser
        # In the full implementation, we'd retrieve the FunctionNode here
        
        # For Phase 1, we can't easily retrieve function nodes from parser storage
        # We'll use a simplified approach where we track function signatures during parsing
        # This is a limitation we'll note and improve in Phase 2
        
        # Return a generic function type for now
        return Type("Function")
    
    fn check_expression(inout self, node_ref: ASTNodeRef) -> Type:
        """Type check an expression and return its type.
        
        Args:
            node_ref: The expression node reference.
            
        Returns:
            The type of the expression.
        """
        let kind = self.parser.node_store.get_node_kind(node_ref)
        
        # Integer literal
        if kind == ASTNodeKind.INTEGER_LITERAL:
            return Type("Int")
        
        # Float literal
        elif kind == ASTNodeKind.FLOAT_LITERAL:
            return Type("Float64")
        
        # String literal
        elif kind == ASTNodeKind.STRING_LITERAL:
            return Type("String")
        
        # Bool literal
        elif kind == ASTNodeKind.BOOL_LITERAL:
            return Type("Bool")
        
        # Identifier - lookup in symbol table
        elif kind == ASTNodeKind.IDENTIFIER_EXPR:
            return self.check_identifier(node_ref)
        
        # Binary expression
        elif kind == ASTNodeKind.BINARY_EXPR:
            return self.check_binary_expr(node_ref)
        
        # Function call
        elif kind == ASTNodeKind.CALL_EXPR:
            return self.check_call_expr(node_ref)
        
        # Unary expression
        elif kind == ASTNodeKind.UNARY_EXPR:
            return self.check_unary_expr(node_ref)
        
        return Type("Unknown")
    
    fn check_identifier(inout self, node_ref: ASTNodeRef) -> Type:
        """Check an identifier and return its type.
        
        Args:
            node_ref: The identifier node reference.
            
        Returns:
            The type of the identifier.
        """
        # Get identifier from parser's identifier nodes list
        if node_ref < 0 or node_ref >= len(self.parser.identifier_nodes):
            self.error("Invalid identifier reference", SourceLocation("", 0, 0))
            return Type("Unknown")
        
        let id_node = self.parser.identifier_nodes[node_ref]
        let symbol_type = self.symbol_table.lookup(id_node.name)
        
        if symbol_type.name == "Unknown":
            self.error("Undefined identifier: " + id_node.name, id_node.location)
        
        return symbol_type
    
    fn check_binary_expr(inout self, node_ref: ASTNodeRef) -> Type:
        """Check a binary expression.
        
        Args:
            node_ref: The binary expression node reference.
            
        Returns:
            The result type of the binary operation.
        """
        # Get binary expression node
        if node_ref < 0 or node_ref >= len(self.parser.binary_expr_nodes):
            self.error("Invalid binary expression reference", SourceLocation("", 0, 0))
            return Type("Unknown")
        
        let binary_node = self.parser.binary_expr_nodes[node_ref]
        
        # Check both operands
        let left_type = self.check_expression(binary_node.left)
        let right_type = self.check_expression(binary_node.right)
        
        # Check type compatibility
        if not left_type.is_compatible_with(right_type):
            self.error(
                "Type mismatch in binary expression: " + left_type.name + 
                " and " + right_type.name,
                binary_node.location
            )
            return Type("Unknown")
        
        # Determine result type based on operator
        if binary_node.operator in ["+", "-", "*", "/", "%"]:
            # Arithmetic operators - return numeric type
            if left_type.is_numeric():
                return left_type
            else:
                self.error("Arithmetic operator requires numeric types", binary_node.location)
                return Type("Unknown")
        
        elif binary_node.operator in ["==", "!=", "<", ">", "<=", ">="]:
            # Comparison operators - return Bool
            return Type("Bool")
        
        elif binary_node.operator in ["and", "or"]:
            # Logical operators - require and return Bool
            if left_type.name == "Bool" and right_type.name == "Bool":
                return Type("Bool")
            else:
                self.error("Logical operator requires Bool types", binary_node.location)
                return Type("Unknown")
        
        return left_type
    
    fn check_call_expr(inout self, node_ref: ASTNodeRef) -> Type:
        """Check a function call expression.
        
        Args:
            node_ref: The call expression node reference.
            
        Returns:
            The return type of the function.
        """
        # Get call expression node
        if node_ref < 0 or node_ref >= len(self.parser.call_expr_nodes):
            self.error("Invalid call expression reference", SourceLocation("", 0, 0))
            return Type("Unknown")
        
        let call_node = self.parser.call_expr_nodes[node_ref]
        
        # Check if function exists
        if not self.symbol_table.is_declared(call_node.callee):
            self.error("Undefined function: " + call_node.callee, call_node.location)
            return Type("Unknown")
        
        # Check argument types
        for i in range(len(call_node.arguments)):
            let arg_ref = call_node.arguments[i]
            _ = self.check_expression(arg_ref)
        
        # For Phase 1, we handle builtin functions specially
        if call_node.callee == "print":
            return Type("NoneType")
        
        # For user-defined functions, we'd look up the signature
        # For now, return Unknown as we need more infrastructure
        return Type("Unknown")
    
    fn check_unary_expr(inout self, node_ref: ASTNodeRef) -> Type:
        """Check a unary expression.
        
        Args:
            node_ref: The unary expression node reference.
            
        Returns:
            The result type.
        """
        # Unary expressions not fully implemented in Phase 1 parser
        return Type("Unknown")
    
    fn check_statement(inout self, node_ref: ASTNodeRef):
        """Type check a statement.
        
        Args:
            node_ref: The statement node reference.
        """
        let kind = self.parser.node_store.get_node_kind(node_ref)
        
        if kind == ASTNodeKind.VAR_DECL:
            self.check_var_decl(node_ref)
        elif kind == ASTNodeKind.RETURN_STMT:
            self.check_return_stmt(node_ref)
        elif kind == ASTNodeKind.EXPR_STMT:
            # Expression statement - just check the expression
            _ = self.check_expression(node_ref)
    
    fn check_var_decl(inout self, node_ref: ASTNodeRef):
        """Check a variable declaration.
        
        Args:
            node_ref: The variable declaration node reference.
        """
        # Get variable declaration node
        if node_ref < 0 or node_ref >= len(self.parser.var_decl_nodes):
            self.error("Invalid variable declaration reference", SourceLocation("", 0, 0))
            return
        
        let var_node = self.parser.var_decl_nodes[node_ref]
        
        # Check if already declared in current scope
        if self.symbol_table.is_declared_in_current_scope(var_node.name):
            self.error("Variable '" + var_node.name + "' already declared", var_node.location)
            return
        
        # Check initializer type
        let init_type = self.check_expression(var_node.initializer)
        
        # Get declared type
        let declared_type = self.type_context.lookup_type(var_node.var_type.name)
        
        # Check type compatibility
        if declared_type.name != "Unknown" and not init_type.is_compatible_with(declared_type):
            self.error(
                "Type mismatch in variable declaration: expected " + declared_type.name +
                ", got " + init_type.name,
                var_node.location
            )
        
        # Use declared type if present, otherwise infer from initializer
        let final_type = declared_type if declared_type.name != "Unknown" else init_type
        
        # Add to symbol table
        if not self.symbol_table.insert(var_node.name, final_type):
            self.error("Failed to declare variable: " + var_node.name, var_node.location)
    
    fn check_return_stmt(inout self, node_ref: ASTNodeRef):
        """Check a return statement.
        
        Args:
            node_ref: The return statement node reference.
        """
        # Get return statement node
        if node_ref < 0 or node_ref >= len(self.parser.return_nodes):
            self.error("Invalid return statement reference", SourceLocation("", 0, 0))
            return
        
        let return_node = self.parser.return_nodes[node_ref]
        
        # Check return value type
        let return_type = Type("NoneType")  # Default for no return value
        if return_node.value != 0:  # 0 means no return value
            return_type = self.check_expression(return_node.value)
        
        # Check against expected function return type
        if not return_type.is_compatible_with(self.current_function_return_type):
            self.error(
                "Return type mismatch: expected " + self.current_function_return_type.name +
                ", got " + return_type.name,
                return_node.location
            )
    
    fn infer_type(inout self, node: ASTNodeRef) -> Type:
        """Infer the type of an expression.
        
        Args:
            node: The expression node reference.
            
        Returns:
            The inferred type.
        """
        # Type inference is handled by check_expression
        return self.check_expression(node)
    
    fn check_ownership(inout self, node: ASTNodeRef) -> Bool:
        """Check ownership rules for a node.
        
        Args:
            node: The node reference to check.
            
        Returns:
            True if ownership rules are satisfied.
        """
        # Ownership checking is Phase 2 - not implemented yet
        # For Phase 1, we assume all ownership rules are satisfied
        return True
    
    fn error(inout self, message: String, location: SourceLocation):
        """Report a type checking error.
        
        Args:
            message: The error message.
            location: The source location of the error.
        """
        let error_msg = str(location) + ": error: " + message
        self.errors.append(error_msg)
    
    fn has_errors(self) -> Bool:
        """Check if any errors occurred during type checking.
        
        Returns:
            True if there are errors.
        """
        return len(self.errors) > 0
    
    fn print_errors(self):
        """Print all type checking errors."""
        for i in range(len(self.errors)):
            print(self.errors[i])
