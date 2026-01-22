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

"""Symbol table for name resolution and scoping.

The symbol table tracks:
- Variable declarations and their types
- Function definitions
- Struct definitions
- Scoping information
"""

from collections import Dict, Optional
from .type_system import Type


struct Symbol:
    """Represents a symbol in the symbol table.
    
    A symbol can be:
    - A variable
    - A function
    - A struct
    - A parameter
    """
    
    var name: String
    var type: Type
    var is_mutable: Bool
    
    fn __init__(inout self, name: String, type: Type, is_mutable: Bool = False):
        """Initialize a symbol.
        
        Args:
            name: The name of the symbol.
            type: The type of the symbol.
            is_mutable: Whether the symbol is mutable.
        """
        self.name = name
        self.type = type
        self.is_mutable = is_mutable


struct SymbolTable:
    """Symbol table for name resolution.
    
    Maintains a hierarchy of scopes for resolving names.
    Supports:
    - Variable lookup
    - Name shadowing
    - Scope management (enter/exit scopes)
    """
    
    var symbols: Dict[String, Symbol]
    var parent: Optional[SymbolTable]
    
    fn __init__(inout self):
        """Initialize an empty symbol table."""
        self.symbols = Dict[String, Symbol]()
        self.parent = None
    
    fn declare(inout self, name: String, symbol: Symbol) raises:
        """Declare a new symbol in the current scope.
        
        Args:
            name: The name of the symbol.
            symbol: The symbol to declare.
            
        Raises:
            Error if the symbol is already declared in this scope.
        """
        if name in self.symbols:
            raise Error("Symbol '" + name + "' is already declared in this scope")
        self.symbols[name] = symbol
    
    fn lookup(self, name: String) -> Optional[Symbol]:
        """Look up a symbol by name.
        
        Searches the current scope and parent scopes.
        
        Args:
            name: The name of the symbol.
            
        Returns:
            The symbol if found, None otherwise.
        """
        if name in self.symbols:
            return self.symbols[name]
        
        # TODO: Check parent scope
        # if self.parent:
        #     return self.parent.lookup(name)
        
        return None
    
    fn enter_scope(inout self) -> SymbolTable:
        """Enter a new scope.
        
        Returns:
            A new symbol table with this one as parent.
        """
        var new_table = SymbolTable()
        # TODO: Set parent reference
        # new_table.parent = self
        return new_table
    
    fn is_declared_in_current_scope(self, name: String) -> Bool:
        """Check if a symbol is declared in the current scope.
        
        Args:
            name: The name of the symbol.
            
        Returns:
            True if declared in current scope (not parent scopes).
        """
        return name in self.symbols
