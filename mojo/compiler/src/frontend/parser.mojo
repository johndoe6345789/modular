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

"""Parser for Mojo source code.

The parser builds an Abstract Syntax Tree (AST) from a stream of tokens.
It handles:
- Module structure
- Function and struct definitions
- Expressions and statements
- Type annotations
- Parameter blocks
- Decorators
"""

from collections import List
from .lexer import Lexer, Token, TokenKind
from .source_location import SourceLocation
from .ast import (
    ModuleNode,
    FunctionNode,
    ParameterNode,
    TypeNode,
    VarDeclNode,
    ReturnStmtNode,
    BinaryExprNode,
    CallExprNode,
    IdentifierExprNode,
    IntegerLiteralNode,
    FloatLiteralNode,
    StringLiteralNode,
    ASTNodeRef,
)


struct AST:
    """Represents the Abstract Syntax Tree for a Mojo module.
    
    The AST is the intermediate representation between parsing and
    semantic analysis. It preserves the structure of the source code.
    """
    
    var root: ModuleNode
    var filename: String
    
    fn __init__(inout self, root: ModuleNode, filename: String):
        """Initialize an AST.
        
        Args:
            root: The root node of the tree (a Module node).
            filename: The source filename.
        """
        self.root = root
        self.filename = filename


struct Parser:
    """Parses Mojo source code into an AST.
    
    The parser uses recursive descent parsing to build the AST from tokens.
    It reports syntax errors with helpful diagnostics.
    """
    
    var lexer: Lexer
    var current_token: Token
    var errors: List[String]
    
    fn __init__(inout self, source: String, filename: String = "<input>"):
        """Initialize the parser with source code.
        
        Args:
            source: The Mojo source code to parse.
            filename: The name of the source file (for error reporting).
        """
        self.lexer = Lexer(source, filename)
        # Get the first token
        self.current_token = self.lexer.next_token()
        self.errors = List[String]()
    
    fn parse(inout self) -> AST:
        """Parse the source code into an AST.
        
        Returns:
            The parsed AST.
        """
        let module = self.parse_module()
        return AST(module, self.lexer.filename)
    
    fn parse_module(inout self) -> ModuleNode:
        """Parse a module (top-level).
        
        Returns:
            The module AST node.
        """
        var module = ModuleNode(SourceLocation(self.lexer.filename, 1, 1))
        
        # Parse top-level declarations
        while self.current_token.kind.kind != TokenKind.EOF:
            # Skip newlines at module level
            if self.current_token.kind.kind == TokenKind.NEWLINE:
                self.advance()
                continue
            
            # Parse function definitions
            if self.current_token.kind.kind == TokenKind.FN:
                let func = self.parse_function()
                # In a real implementation, we would add the function to the module
                # module.add_declaration(func)
            else:
                self.error("Expected function or struct definition")
                self.advance()  # Skip the problematic token
        
        return module
    
    fn parse_function(inout self) -> FunctionNode:
        """Parse a function definition.
        
        Returns:
            The function AST node.
        """
        let start_location = self.current_token.location
        
        # Expect 'fn' keyword
        if not self.expect(TokenKind(TokenKind.FN)):
            self.error("Expected 'fn'")
            return FunctionNode("error", start_location)
        
        # Parse function name
        if self.current_token.kind.kind != TokenKind.IDENTIFIER:
            self.error("Expected function name")
            return FunctionNode("error", start_location)
        
        let name = self.current_token.text
        self.advance()
        
        var func = FunctionNode(name, start_location)
        
        # Parse parameter list
        if not self.expect(TokenKind(TokenKind.LEFT_PAREN)):
            self.error("Expected '('")
            return func
        
        # Parse parameters (simplified - no implementation for now)
        # TODO: Implement parameter parsing
        
        if not self.expect(TokenKind(TokenKind.RIGHT_PAREN)):
            self.error("Expected ')'")
            return func
        
        # Parse optional return type
        if self.current_token.kind.kind == TokenKind.ARROW:
            self.advance()
            # TODO: Parse return type
        
        # Expect colon
        if not self.expect(TokenKind(TokenKind.COLON)):
            self.error("Expected ':'")
            return func
        
        # Parse function body
        # TODO: Implement statement parsing for body
        
        return func
    
    fn parse_struct(inout self) -> FunctionNode:
        """Parse a struct definition.
        
        Returns:
            The struct AST node (placeholder - returns function for now).
        """
        # TODO: Implement struct parsing
        # struct Name[params]: fields and methods
        let location = self.current_token.location
        return FunctionNode("placeholder", location)
    
    fn parse_statement(inout self) -> ASTNodeRef:
        """Parse a statement.
        
        Returns:
            The statement AST node reference.
        """
        # Return statement
        if self.current_token.kind.kind == TokenKind.RETURN:
            return self.parse_return_statement()
        
        # Variable declaration
        if self.current_token.kind.kind == TokenKind.VAR or self.current_token.kind.kind == TokenKind.LET:
            return self.parse_var_declaration()
        
        # Expression statement (e.g., function call)
        return self.parse_expression_statement()
    
    fn parse_return_statement(inout self) -> ASTNodeRef:
        """Parse a return statement.
        
        Returns:
            The return statement node reference.
        """
        let location = self.current_token.location
        self.advance()  # Skip 'return'
        
        # Parse optional return value
        var value: ASTNodeRef = 0  # Placeholder for None
        if self.current_token.kind.kind != TokenKind.NEWLINE:
            value = self.parse_expression()
        
        # In a real implementation, would create and return ReturnStmtNode
        return 0  # Placeholder
    
    fn parse_var_declaration(inout self) -> ASTNodeRef:
        """Parse a variable declaration.
        
        Returns:
            The variable declaration node reference.
        """
        let is_var = self.current_token.kind.kind == TokenKind.VAR
        self.advance()  # Skip 'var' or 'let'
        
        # Parse variable name
        if self.current_token.kind.kind != TokenKind.IDENTIFIER:
            self.error("Expected variable name")
            return 0
        
        let name = self.current_token.text
        self.advance()
        
        # Parse optional type annotation
        if self.current_token.kind.kind == TokenKind.COLON:
            self.advance()
            # TODO: Parse type
        
        # Parse initializer
        if self.current_token.kind.kind == TokenKind.EQUAL:
            self.advance()
            let init = self.parse_expression()
            # TODO: Create VarDeclNode
        
        return 0  # Placeholder
    
    fn parse_expression_statement(inout self) -> ASTNodeRef:
        """Parse an expression statement.
        
        Returns:
            The expression node reference.
        """
        return self.parse_expression()
    
    fn parse_expression(inout self) -> ASTNodeRef:
        """Parse an expression.
        
        Returns:
            The expression AST node reference.
        """
        # For now, just parse primary expressions
        # TODO: Implement full expression parsing with precedence
        return self.parse_primary_expression()
    
    fn parse_primary_expression(inout self) -> ASTNodeRef:
        """Parse a primary expression (literals, identifiers, calls).
        
        Returns:
            The expression node reference.
        """
        # Integer literal
        if self.current_token.kind.kind == TokenKind.INTEGER_LITERAL:
            let value = self.current_token.text
            let location = self.current_token.location
            self.advance()
            # TODO: Create and return IntegerLiteralNode
            return 0  # Placeholder
        
        # Float literal
        if self.current_token.kind.kind == TokenKind.FLOAT_LITERAL:
            let value = self.current_token.text
            let location = self.current_token.location
            self.advance()
            # TODO: Create and return FloatLiteralNode
            return 0  # Placeholder
        
        # String literal
        if self.current_token.kind.kind == TokenKind.STRING_LITERAL:
            let value = self.current_token.text
            let location = self.current_token.location
            self.advance()
            # TODO: Create and return StringLiteralNode
            return 0  # Placeholder
        
        # Identifier or function call
        if self.current_token.kind.kind == TokenKind.IDENTIFIER:
            let name = self.current_token.text
            let location = self.current_token.location
            self.advance()
            
            # Check for function call
            if self.current_token.kind.kind == TokenKind.LEFT_PAREN:
                return self.parse_call_expression(name, location)
            
            # Just an identifier
            # TODO: Create and return IdentifierExprNode
            return 0  # Placeholder
        
        # Parenthesized expression
        if self.current_token.kind.kind == TokenKind.LEFT_PAREN:
            self.advance()
            let expr = self.parse_expression()
            if not self.expect(TokenKind(TokenKind.RIGHT_PAREN)):
                self.error("Expected ')'")
            return expr
        
        self.error("Expected expression")
        return 0  # Placeholder
    
    fn parse_call_expression(inout self, callee: String, location: SourceLocation) -> ASTNodeRef:
        """Parse a function call expression.
        
        Args:
            callee: The function name being called.
            location: Source location of the call.
            
        Returns:
            The call expression node reference.
        """
        self.advance()  # Skip '('
        
        # Parse arguments
        # TODO: Implement argument list parsing
        # For now, just parse until ')'
        
        while self.current_token.kind.kind != TokenKind.RIGHT_PAREN and self.current_token.kind.kind != TokenKind.EOF:
            let arg = self.parse_expression()
            
            # Check for comma
            if self.current_token.kind.kind == TokenKind.COMMA:
                self.advance()
            elif self.current_token.kind.kind != TokenKind.RIGHT_PAREN:
                break
        
        if not self.expect(TokenKind(TokenKind.RIGHT_PAREN)):
            self.error("Expected ')'")
        
        # TODO: Create and return CallExprNode
        return 0  # Placeholder
    
    fn parse_type(inout self) -> TypeNode:
        """Parse a type annotation.
        
        Returns:
            The type AST node.
        """
        if self.current_token.kind.kind != TokenKind.IDENTIFIER:
            self.error("Expected type name")
            return TypeNode("Error", self.current_token.location)
        
        let type_name = self.current_token.text
        let location = self.current_token.location
        self.advance()
        
        # TODO: Handle parametric types like List[Int]
        
        return TypeNode(type_name, location)
    
    fn expect(inout self, kind: TokenKind) -> Bool:
        """Check if current token matches expected kind and advance.
        
        Args:
            kind: The expected token kind.
            
        Returns:
            True if matched, False otherwise.
        """
        if self.current_token.kind.kind == kind.kind:
            self.advance()
            return True
        return False
    
    fn advance(inout self):
        """Advance to the next token."""
        self.current_token = self.lexer.next_token()
    
    fn error(inout self, message: String):
        """Report a parse error.
        
        Args:
            message: The error message.
        """
        let loc = self.current_token.location
        let error_msg = str(loc) + ": error: " + message
        self.errors.append(error_msg)
    
    fn has_errors(self) -> Bool:
        """Check if any errors were encountered.
        
        Returns:
            True if there were errors, False otherwise.
        """
        return len(self.errors) > 0
