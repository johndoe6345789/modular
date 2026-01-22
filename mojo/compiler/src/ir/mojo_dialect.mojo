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

"""Mojo dialect definition for MLIR.

This module defines the Mojo-specific MLIR dialect that represents
Mojo language semantics:

Operations include:
- mojo.func: Function definition
- mojo.struct: Struct type definition
- mojo.trait: Trait definition
- mojo.own: Ownership operation
- mojo.borrow: Borrow operation
- mojo.mut_borrow: Mutable borrow operation
- mojo.move: Move operation
- mojo.copy: Copy operation
- mojo.parametric_call: Parametric function call
- mojo.trait_call: Trait method call

Types include:
- !mojo.value<T>: Owned value type
- !mojo.ref<T>: Borrowed reference type
- !mojo.mut_ref<T>: Mutable borrowed reference type
- !mojo.struct<name, fields>: Struct type
- !mojo.trait<name>: Trait type
"""


struct MojoDialect:
    """Represents the Mojo MLIR dialect.
    
    This dialect extends MLIR with Mojo-specific operations and types.
    It bridges the gap between high-level Mojo semantics and lower-level
    MLIR/LLVM representations.
    """
    
    var name: String
    
    fn __init__(inout self):
        """Initialize the Mojo dialect."""
        self.name = "mojo"
    
    fn register_operations(inout self):
        """Register all Mojo dialect operations.
        
        This includes:
        - Memory operations (own, borrow, move, copy)
        - Function operations
        - Struct operations
        - Trait operations
        """
        # TODO: Register MLIR operations
        pass
    
    fn register_types(inout self):
        """Register all Mojo dialect types.
        
        This includes:
        - Value types
        - Reference types
        - Struct types
        - Trait types
        """
        # TODO: Register MLIR types
        pass


struct MojoOperation:
    """Base struct for Mojo MLIR operations."""
    
    var name: String
    var operands: List[String]
    var results: List[String]
    
    fn __init__(inout self, name: String):
        """Initialize a Mojo operation.
        
        Args:
            name: The name of the operation (e.g., "mojo.own").
        """
        self.name = name
        self.operands = List[String]()
        self.results = List[String]()


struct MojoType:
    """Base struct for Mojo MLIR types."""
    
    var name: String
    
    fn __init__(inout self, name: String):
        """Initialize a Mojo type.
        
        Args:
            name: The name of the type (e.g., "!mojo.value<Int>").
        """
        self.name = name
