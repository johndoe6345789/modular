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

"""Lexer for Mojo source code.

The lexer is responsible for tokenizing Mojo source code into a stream of tokens.
It handles:
- Keywords (fn, struct, var, def, etc.)
- Identifiers
- Literals (integers, floats, strings)
- Operators and punctuation
- Comments
- Whitespace (for indentation-based syntax)
"""

from .source_location import SourceLocation


@value
struct TokenKind:
    """Represents the kind of a token."""
    
    # Keywords
    alias FN = 0
    alias STRUCT = 1
    alias TRAIT = 2
    alias VAR = 3
    alias DEF = 4
    alias IF = 5
    alias ELSE = 6
    alias ELIF = 7
    alias WHILE = 8
    alias FOR = 9
    alias IN = 10
    alias RETURN = 11
    alias BREAK = 12
    alias CONTINUE = 13
    alias PASS = 14
    alias IMPORT = 15
    alias FROM = 16
    alias AS = 17
    alias ALIAS = 18
    alias LET = 19
    
    # Literals
    alias IDENTIFIER = 100
    alias INTEGER_LITERAL = 101
    alias FLOAT_LITERAL = 102
    alias STRING_LITERAL = 103
    alias BOOL_LITERAL = 104
    
    # Operators
    alias PLUS = 200
    alias MINUS = 201
    alias STAR = 202
    alias SLASH = 203
    alias PERCENT = 204
    alias DOUBLE_STAR = 205
    alias EQUAL = 206
    alias DOUBLE_EQUAL = 207
    alias NOT_EQUAL = 208
    alias LESS = 209
    alias GREATER = 210
    alias LESS_EQUAL = 211
    alias GREATER_EQUAL = 212
    alias AMPERSAND = 213
    alias PIPE = 214
    alias CARET = 215
    alias TILDE = 216
    alias DOUBLE_AMPERSAND = 217
    alias DOUBLE_PIPE = 218
    alias EXCLAMATION = 219
    alias ARROW = 220
    
    # Punctuation
    alias LEFT_PAREN = 300
    alias RIGHT_PAREN = 301
    alias LEFT_BRACKET = 302
    alias RIGHT_BRACKET = 303
    alias LEFT_BRACE = 304
    alias RIGHT_BRACE = 305
    alias COMMA = 306
    alias COLON = 307
    alias SEMICOLON = 308
    alias DOT = 309
    alias AT = 310
    alias QUESTION = 311
    
    # Special
    alias NEWLINE = 400
    alias INDENT = 401
    alias DEDENT = 402
    alias EOF = 403
    alias ERROR = 404
    
    var kind: Int
    
    fn __init__(inout self, kind: Int):
        self.kind = kind


struct Token:
    """Represents a lexical token."""
    
    var kind: TokenKind
    var text: String
    var location: SourceLocation
    
    fn __init__(inout self, kind: TokenKind, text: String, location: SourceLocation):
        self.kind = kind
        self.text = text
        self.location = location


struct Lexer:
    """Tokenizes Mojo source code.
    
    The lexer processes source text character by character and produces tokens.
    It handles indentation-based syntax similar to Python.
    """
    
    var source: String
    var position: Int
    var line: Int
    var column: Int
    var filename: String
    
    fn __init__(inout self, source: String, filename: String = "<input>"):
        """Initialize the lexer with source code.
        
        Args:
            source: The Mojo source code to tokenize.
            filename: The name of the source file (for error reporting).
        """
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.filename = filename
    
    fn next_token(inout self) -> Token:
        """Get the next token from the source.
        
        Returns:
            The next token in the source stream.
        """
        # TODO: Implement tokenization logic
        # This is a placeholder implementation
        return Token(
            TokenKind(TokenKind.EOF),
            "",
            SourceLocation(self.filename, self.line, self.column)
        )
    
    fn peek_char(self) -> String:
        """Peek at the current character without consuming it.
        
        Returns:
            The current character, or empty string if at EOF.
        """
        # TODO: Implement character peeking
        return ""
    
    fn advance(inout self):
        """Advance to the next character in the source."""
        # TODO: Implement character advancement with line/column tracking
        self.position += 1
        self.column += 1
    
    fn skip_whitespace(inout self):
        """Skip whitespace characters (except newlines for indentation tracking)."""
        # TODO: Implement whitespace skipping
        pass
    
    fn skip_comment(inout self):
        """Skip a comment (# to end of line)."""
        # TODO: Implement comment skipping
        pass
    
    fn read_identifier(inout self) -> String:
        """Read an identifier or keyword.
        
        Returns:
            The identifier text.
        """
        # TODO: Implement identifier reading
        return ""
    
    fn read_number(inout self) -> Token:
        """Read a numeric literal (integer or float).
        
        Returns:
            A token representing the number.
        """
        # TODO: Implement number reading
        return Token(
            TokenKind(TokenKind.INTEGER_LITERAL),
            "0",
            SourceLocation(self.filename, self.line, self.column)
        )
    
    fn read_string(inout self) -> String:
        """Read a string literal.
        
        Returns:
            The string content (without quotes).
        """
        # TODO: Implement string reading with escape sequences
        return ""
    
    fn is_keyword(self, text: String) -> Bool:
        """Check if a string is a keyword.
        
        Args:
            text: The text to check.
            
        Returns:
            True if the text is a keyword, False otherwise.
        """
        # TODO: Implement keyword checking
        return False
    
    fn keyword_kind(self, text: String) -> TokenKind:
        """Get the token kind for a keyword.
        
        Args:
            text: The keyword text.
            
        Returns:
            The corresponding TokenKind.
        """
        # TODO: Implement keyword to TokenKind mapping
        return TokenKind(TokenKind.IDENTIFIER)
