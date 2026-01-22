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

"""Type system for the Mojo compiler.

This module defines the type system including:
- Builtin types (Int, Float, Bool, String, etc.)
- User-defined types (structs)
- Parametric types and generics
- Trait types
- Reference types
"""


struct Type:
    """Represents a type in the Mojo type system.
    
    Types can be:
    - Builtin types (Int, Float64, Bool, String)
    - User-defined types (struct definitions)
    - Parametric types (List[T], Dict[K, V])
    - Trait types
    - Reference types (owned, borrowed, mutable)
    """
    
    var name: String
    var is_parametric: Bool
    
    fn __init__(inout self, name: String, is_parametric: Bool = False):
        """Initialize a type.
        
        Args:
            name: The name of the type.
            is_parametric: Whether this is a parametric type.
        """
        self.name = name
        self.is_parametric = is_parametric
    
    fn is_builtin(self) -> Bool:
        """Check if this is a builtin type.
        
        Returns:
            True if this is a builtin type.
        """
        # TODO: Implement builtin type checking
        return self.name in ["Int", "Float64", "Bool", "String"]
    
    fn is_compatible_with(self, other: Type) -> Bool:
        """Check if this type is compatible with another type.
        
        Args:
            other: The other type to check compatibility with.
            
        Returns:
            True if the types are compatible.
        """
        # TODO: Implement type compatibility checking
        # This should handle subtyping, trait conformance, etc.
        return self.name == other.name


struct TypeContext:
    """Context for type checking and type inference.
    
    Maintains information about:
    - Declared types
    - Type parameters
    - Trait implementations
    """
    
    var types: Dict[String, Type]
    
    fn __init__(inout self):
        """Initialize a type context with builtin types."""
        self.types = Dict[String, Type]()
        # Register builtin types
        self.register_builtin_types()
    
    fn register_builtin_types(inout self):
        """Register all builtin types."""
        # TODO: Register all builtin types
        self.types["Int"] = Type("Int")
        self.types["Float64"] = Type("Float64")
        self.types["Bool"] = Type("Bool")
        self.types["String"] = Type("String")
    
    fn register_type(inout self, name: String, type: Type):
        """Register a user-defined type.
        
        Args:
            name: The name of the type.
            type: The type to register.
        """
        self.types[name] = type
    
    fn lookup_type(self, name: String) -> Type:
        """Look up a type by name.
        
        Args:
            name: The name of the type.
            
        Returns:
            The type, or raises an error if not found.
        """
        # TODO: Implement type lookup with error handling
        return self.types.get(name, Type("Unknown"))
    
    fn check_trait_conformance(self, type: Type, trait_name: String) -> Bool:
        """Check if a type conforms to a trait.
        
        Args:
            type: The type to check.
            trait_name: The name of the trait.
            
        Returns:
            True if the type conforms to the trait.
        """
        # TODO: Implement trait conformance checking
        return False
