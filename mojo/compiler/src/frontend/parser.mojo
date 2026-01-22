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

from .lexer import Lexer, Token, TokenKind
from .source_location import SourceLocation


trait ASTNode:
    """Base trait for all AST nodes.
    
    All AST nodes should implement this trait to support visitor pattern.
    """
    
    fn location(self) -> SourceLocation:
        """Get the source location of this node.
        
        Returns:
            The source location.
        """
        ...


struct AST:
    """Represents the Abstract Syntax Tree for a Mojo module.
    
    The AST is the intermediate representation between parsing and
    semantic analysis. It preserves the structure of the source code.
    """
    
    var root: ASTNode
    var filename: String
    
    fn __init__(inout self, root: ASTNode, filename: String):
        """Initialize an AST.
        
        Args:
            root: The root node of the tree (typically a Module node).
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
        # TODO: Implement parsing logic
        # This should parse a complete module
        return AST(self.parse_module(), self.lexer.filename)
    
    fn parse_module(inout self) -> ASTNode:
        """Parse a module (top-level).
        
        Returns:
            The module AST node.
        """
        # TODO: Implement module parsing
        # A module consists of imports, functions, structs, etc.
        pass
    
    fn parse_function(inout self) -> ASTNode:
        """Parse a function definition.
        
        Returns:
            The function AST node.
        """
        # TODO: Implement function parsing
        # fn name[params](args) -> return_type: body
        pass
    
    fn parse_struct(inout self) -> ASTNode:
        """Parse a struct definition.
        
        Returns:
            The struct AST node.
        """
        # TODO: Implement struct parsing
        # struct Name[params]: fields and methods
        pass
    
    fn parse_statement(inout self) -> ASTNode:
        """Parse a statement.
        
        Returns:
            The statement AST node.
        """
        # TODO: Implement statement parsing
        # var, let, return, if, while, for, etc.
        pass
    
    fn parse_expression(inout self) -> ASTNode:
        """Parse an expression.
        
        Returns:
            The expression AST node.
        """
        # TODO: Implement expression parsing with proper precedence
        # Binary ops, unary ops, calls, indexing, etc.
        pass
    
    fn parse_type(inout self) -> ASTNode:
        """Parse a type annotation.
        
        Returns:
            The type AST node.
        """
        # TODO: Implement type parsing
        # Int, String, List[T], MyStruct[T, U], etc.
        pass
    
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
