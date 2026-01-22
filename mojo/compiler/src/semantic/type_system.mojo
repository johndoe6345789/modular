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

from collections import Dict, Optional, List


struct FieldInfo:
    """Information about a struct field."""
    
    var name: String
    var field_type: Type
    
    fn __init__(inout self, name: String, field_type: Type):
        """Initialize field info.
        
        Args:
            name: The field name.
            field_type: The field type.
        """
        self.name = name
        self.field_type = field_type


struct MethodInfo:
    """Information about a struct method."""
    
    var name: String
    var parameter_types: List[Type]
    var return_type: Type
    
    fn __init__(inout self, name: String, return_type: Type):
        """Initialize method info.
        
        Args:
            name: The method name.
            return_type: The method return type.
        """
        self.name = name
        self.parameter_types = List[Type]()
        self.return_type = return_type


struct StructInfo:
    """Information about a struct type."""
    
    var name: String
    var fields: List[FieldInfo]
    var methods: List[MethodInfo]
    
    fn __init__(inout self, name: String):
        """Initialize struct info.
        
        Args:
            name: The struct name.
        """
        self.name = name
        self.fields = List[FieldInfo]()
        self.methods = List[MethodInfo]()
    
    fn add_field(inout self, name: String, field_type: Type):
        """Add a field to the struct.
        
        Args:
            name: The field name.
            field_type: The field type.
        """
        self.fields.append(FieldInfo(name, field_type))
    
    fn add_method(inout self, name: String, return_type: Type):
        """Add a method to the struct.
        
        Args:
            name: The method name.
            return_type: The method return type.
        """
        self.methods.append(MethodInfo(name, return_type))
    
    fn get_field_type(self, field_name: String) -> Type:
        """Get the type of a field by name.
        
        Args:
            field_name: The name of the field.
            
        Returns:
            The field type, or Unknown if not found.
        """
        for i in range(len(self.fields)):
            if self.fields[i].name == field_name:
                return self.fields[i].field_type
        return Type("Unknown")
    
    fn has_method(self, method_name: String) -> Bool:
        """Check if the struct has a method.
        
        Args:
            method_name: The method name.
            
        Returns:
            True if the method exists.
        """
        for i in range(len(self.methods)):
            if self.methods[i].name == method_name:
                return True
        return False
    
    fn get_method(self, method_name: String) -> MethodInfo:
        """Get method info by name.
        
        Args:
            method_name: The method name.
            
        Returns:
            The method info, or a dummy method if not found.
        """
        for i in range(len(self.methods)):
            if self.methods[i].name == method_name:
                return self.methods[i]
        return MethodInfo("unknown", Type("Unknown"))


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
    var is_reference: Bool
    var is_struct: Bool  # Track if this is a struct type
    var element_type: Optional[String]  # For containers like List[T]
    
    fn __init__(inout self, name: String, is_parametric: Bool = False, is_reference: Bool = False, is_struct: Bool = False):
        """Initialize a type.
        
        Args:
            name: The name of the type.
            is_parametric: Whether this is a parametric type.
            is_reference: Whether this is a reference type.
            is_struct: Whether this is a struct type.
        """
        self.name = name
        self.is_parametric = is_parametric
        self.is_reference = is_reference
        self.is_struct = is_struct
        self.element_type = None
    
    fn is_builtin(self) -> Bool:
        """Check if this is a builtin type.
        
        Returns:
            True if this is a builtin type.
        """
        let builtins = ["Int", "Float64", "Float32", "Bool", "String", "UInt8", "UInt16", "UInt32", "UInt64", "Int8", "Int16", "Int32", "Int64", "NoneType"]
        return self.name in builtins
    
    fn is_numeric(self) -> Bool:
        """Check if this is a numeric type.
        
        Returns:
            True if this is a numeric type.
        """
        let numeric = ["Int", "Float64", "Float32", "UInt8", "UInt16", "UInt32", "UInt64", "Int8", "Int16", "Int32", "Int64"]
        return self.name in numeric
    
    fn is_integer(self) -> Bool:
        """Check if this is an integer type.
        
        Returns:
            True if this is an integer type.
        """
        let integers = ["Int", "UInt8", "UInt16", "UInt32", "UInt64", "Int8", "Int16", "Int32", "Int64"]
        return self.name in integers
    
    fn is_float(self) -> Bool:
        """Check if this is a floating point type.
        
        Returns:
            True if this is a floating point type.
        """
        return self.name in ["Float32", "Float64"]
    
    fn is_compatible_with(self, other: Type) -> Bool:
        """Check if this type is compatible with another type.
        
        Args:
            other: The other type to check compatibility with.
            
        Returns:
            True if the types are compatible.
        """
        # Exact match
        if self.name == other.name:
            return True
        
        # Numeric type promotions
        if self.is_numeric() and other.is_numeric():
            # Allow implicit promotion from smaller to larger types
            if self.is_integer() and other.is_integer():
                return True  # Simplified: allow any integer promotion
            if self.is_integer() and other.is_float():
                return True  # Int to Float promotion
        
        # Unknown type is compatible with anything (for inference)
        if self.name == "Unknown" or other.name == "Unknown":
            return True
        
        return False
    
    fn __eq__(self, other: Type) -> Bool:
        """Check equality with another type."""
        return self.name == other.name


struct TypeContext:
    """Context for type checking and type inference.
    
    Maintains information about:
    - Declared types
    - Type parameters
    - Trait implementations
    - Struct definitions
    """
    
    var types: Dict[String, Type]
    var structs: Dict[String, StructInfo]  # Store struct definitions
    
    fn __init__(inout self):
        """Initialize a type context with builtin types."""
        self.types = Dict[String, Type]()
        self.structs = Dict[String, StructInfo]()
        # Register builtin types
        self.register_builtin_types()
    
    fn register_builtin_types(inout self):
        """Register all builtin types."""
        # Integer types
        self.types["Int"] = Type("Int")
        self.types["Int8"] = Type("Int8")
        self.types["Int16"] = Type("Int16")
        self.types["Int32"] = Type("Int32")
        self.types["Int64"] = Type("Int64")
        self.types["UInt8"] = Type("UInt8")
        self.types["UInt16"] = Type("UInt16")
        self.types["UInt32"] = Type("UInt32")
        self.types["UInt64"] = Type("UInt64")
        
        # Floating point types
        self.types["Float32"] = Type("Float32")
        self.types["Float64"] = Type("Float64")
        
        # Boolean and String
        self.types["Bool"] = Type("Bool")
        self.types["String"] = Type("String")
        self.types["StringLiteral"] = Type("StringLiteral")
        
        # Special types
        self.types["NoneType"] = Type("NoneType")
        self.types["Unknown"] = Type("Unknown")
    
    fn register_type(inout self, name: String, type: Type):
        """Register a user-defined type.
        
        Args:
            name: The name of the type.
            type: The type to register.
        """
        self.types[name] = type
    
    fn register_struct(inout self, struct_info: StructInfo):
        """Register a struct type.
        
        Args:
            struct_info: The struct information to register.
        """
        self.structs[struct_info.name] = struct_info
        # Also register as a type
        self.types[struct_info.name] = Type(struct_info.name, is_struct=True)
    
    fn lookup_struct(self, name: String) -> StructInfo:
        """Look up a struct by name.
        
        Args:
            name: The name of the struct.
            
        Returns:
            The struct info, or an empty struct if not found.
        """
        return self.structs.get(name, StructInfo("Unknown"))
    
    fn is_struct(self, name: String) -> Bool:
        """Check if a type is a struct.
        
        Args:
            name: The type name.
            
        Returns:
            True if the type is a struct.
        """
        return name in self.structs
    
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
