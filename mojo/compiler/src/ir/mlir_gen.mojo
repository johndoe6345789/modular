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

"""MLIR code generation from typed AST.

This module lowers the typed AST to MLIR representation using
the Mojo dialect and standard MLIR dialects (arith, scf, func, etc.).
"""

from collections import List
from ..frontend.parser import AST
from ..frontend.ast import ModuleNode, ASTNodeRef
from .mojo_dialect import MojoDialect


struct MLIRGenerator:
    """Generates MLIR code from a typed AST.
    
    The generator walks the AST and emits MLIR operations and types.
    It uses:
    - Mojo dialect for Mojo-specific operations
    - Standard MLIR dialects (arith, scf, func, cf) for common operations
    """
    
    var dialect: MojoDialect
    var output: String
    
    fn __init__(inout self):
        """Initialize the MLIR generator."""
        self.dialect = MojoDialect()
        self.output = ""
    
    fn generate(inout self, ast: AST) -> String:
        """Generate MLIR code from an AST.
        
        Args:
            ast: The typed AST to convert to MLIR.
            
        Returns:
            The generated MLIR code as a string.
        """
        # TODO: Implement MLIR generation
        # Generate module header
        self.emit("module {")
        
        # Visit all nodes and generate MLIR
        self.generate_node(ast.root)
        
        self.emit("}")
        return self.output
    
    fn generate_node(inout self, node: ModuleNode):
        """Generate MLIR for a module node.
        
        Args:
            node: The module node to generate MLIR for.
        """
        # TODO: Implement node-specific MLIR generation
        # For now, iterate through declarations
        for decl in node.declarations:
            # Generate MLIR for each declaration
            # self.generate_function(decl)
            pass
    
    fn generate_function(inout self, node: ASTNodeRef):
        """Generate MLIR for a function definition.
        
        Args:
            node: Reference to the function node.
        """
        # TODO: Implement function MLIR generation
        # Example output:
        # func.func @function_name(%arg0: i64) -> i64 {
        #   ...
        # }
        pass
    
    fn generate_expression(inout self, node: ASTNodeRef) -> String:
        """Generate MLIR for an expression.
        
        Args:
            node: Reference to the expression node.
            
        Returns:
            The MLIR value name (e.g., "%0", "%result").
        """
        # TODO: Implement expression MLIR generation
        return "%0"
    
    fn generate_statement(inout self, node: ASTNodeRef):
        """Generate MLIR for a statement.
        
        Args:
            node: Reference to the statement node.
        """
        # TODO: Implement statement MLIR generation
        pass
    
    fn emit(inout self, code: String):
        """Emit MLIR code to the output.
        
        Args:
            code: The MLIR code to emit.
        """
        self.output += code + "\n"
    
    fn emit_ownership_op(inout self, op: String, value: String) -> String:
        """Emit an ownership operation (own, borrow, move, copy).
        
        Args:
            op: The operation name (own, borrow, move, copy).
            value: The value to operate on.
            
        Returns:
            The result value name.
        """
        # TODO: Implement ownership operation emission
        # Example:
        # %owned = mojo.own %value : !mojo.value<String>
        return "%owned"
    
    fn emit_type(self, type_name: String) -> String:
        """Convert a Mojo type to MLIR type syntax.
        
        Args:
            type_name: The Mojo type name.
            
        Returns:
            The MLIR type representation.
        """
        # Map Mojo types to MLIR types
        if type_name == "Int" or type_name == "Int64":
            return "i64"
        elif type_name == "Int32":
            return "i32"
        elif type_name == "Int16":
            return "i16"
        elif type_name == "Int8":
            return "i8"
        elif type_name == "UInt64":
            return "i64"
        elif type_name == "UInt32":
            return "i32"
        elif type_name == "UInt16":
            return "i16"
        elif type_name == "UInt8":
            return "i8"
        elif type_name == "Float64":
            return "f64"
        elif type_name == "Float32":
            return "f32"
        elif type_name == "Bool":
            return "i1"
        elif type_name == "String":
            return "!mojo.value<String>"
        elif type_name == "NoneType" or type_name == "None":
            return "!mojo.none"
        else:
            # Custom types
            return "!mojo.value<" + type_name + ">"
    
    fn generate_builtin_call(inout self, function_name: String, args: List[String]) -> String:
        """Generate MLIR for builtin function calls like print.
        
        Args:
            function_name: The builtin function name.
            args: The argument value names.
            
        Returns:
            The result value name (if any).
        """
        if function_name == "print":
            # Generate print call
            self.emit("  mojo.print " + ", ".join(args))
            return ""
        else:
            # Generic builtin call
            self.emit("  mojo.call_builtin @" + function_name + "(" + ", ".join(args) + ")")
            return "%result"
