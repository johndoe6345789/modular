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

from collections import Dict, List
from ..frontend.parser import AST, Parser
from ..frontend.ast import (
    ModuleNode,
    FunctionNode,
    ParameterNode,
    ReturnStmtNode,
    VarDeclNode,
    BinaryExprNode,
    CallExprNode,
    IdentifierExprNode,
    IntegerLiteralNode,
    FloatLiteralNode,
    StringLiteralNode,
    ASTNodeRef,
    ASTNodeKind,
)
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
    var parser: Parser  # Reference to parser for node access
    var ssa_counter: Int  # Counter for SSA value names
    var indent_level: Int  # Current indentation level
    var identifier_map: Dict[String, String]  # Maps identifier names to SSA values
    
    fn __init__(inout self, owned parser: Parser):
        """Initialize the MLIR generator.
        
        Args:
            parser: Parser containing the AST nodes.
        """
        self.dialect = MojoDialect()
        self.output = ""
        self.parser = parser^
        self.ssa_counter = 0
        self.indent_level = 0
        self.identifier_map = Dict[String, String]()
    
    fn next_ssa_value(inout self) -> String:
        """Generate the next SSA value name.
        
        Returns:
            A unique SSA value name like "%0", "%1", etc.
        """
        let result = "%" + str(self.ssa_counter)
        self.ssa_counter += 1
        return result
    
    fn get_indent(self) -> String:
        """Get the current indentation string.
        
        Returns:
            A string of spaces for the current indent level.
        """
        var result = ""
        for i in range(self.indent_level):
            result += "  "
        return result
    
    fn generate_module_with_functions(inout self, functions: List[FunctionNode]) -> String:
        """Generate complete MLIR module from a list of functions.
        
        Args:
            functions: List of function nodes to generate.
            
        Returns:
            The generated MLIR module text.
        """
        self.output = ""
        self.ssa_counter = 0
        self.indent_level = 0
        
        # Emit module header
        self.emit("module {")
        self.indent_level += 1
        
        # Generate each function
        for func in functions:
            self.generate_function_direct(func[])
            self.emit("")  # Blank line between functions
        
        self.indent_level -= 1
        self.emit("}")
        
        return self.output
    
    fn generate_function_direct(inout self, func: FunctionNode):
        """Generate MLIR for a function definition (direct API).
        
        Args:
            func: The function node to generate.
        """
        # Reset SSA counter and identifier map for each function
        self.ssa_counter = 0
        self.identifier_map = Dict[String, String]()
        
        let indent = self.get_indent()
        
        # Build function signature
        var signature = indent + "func.func @" + func.name + "("
        
        # Add parameters and track in identifier map
        for i in range(len(func.parameters)):
            if i > 0:
                signature += ", "
            let param = func.parameters[i]
            let param_type = self.emit_type(param.param_type.name)
            let arg_name = "%arg" + str(i)
            signature += arg_name + ": " + param_type
            # Track parameter name to SSA value mapping
            self.identifier_map[param.name] = arg_name
        
        signature += ")"
        
        # Add return type if not None
        if func.return_type.name != "None" and func.return_type.name != "NoneType":
            signature += " -> " + self.emit_type(func.return_type.name)
        
        signature += " {"
        self.emit(signature)
        
        # Generate function body
        self.indent_level += 1
        for stmt_ref in func.body:
            self.generate_statement(stmt_ref)
        
        self.indent_level -= 1
        self.emit(indent + "}")
    
    fn generate_module(inout self, module: ModuleNode) -> String:
        """Generate complete MLIR module from ModuleNode.
        
        Args:
            module: The module node to generate MLIR for.
            
        Returns:
            The generated MLIR module text.
        """
        self.output = ""
        self.ssa_counter = 0
        self.indent_level = 0
        
        # Emit module header
        self.emit("module {")
        self.indent_level += 1
        
        # Generate each function declaration
        for decl_ref in module.declarations:
            self.generate_function(decl_ref)
            self.emit("")  # Blank line between functions
        
        self.indent_level -= 1
        self.emit("}")
        
        return self.output
    
    fn generate_function(inout self, node_ref: ASTNodeRef):
        """Generate MLIR for a function definition.
        
        Args:
            node_ref: Reference to the function node.
        """
        # Note: This is a stub for the module-based API
        # In Phase 1, we use generate_function_direct() instead
        let indent = self.get_indent()
        self.emit(indent + "func.func @placeholder() {")
        self.indent_level += 1
        self.emit(self.get_indent() + "return")
        self.indent_level -= 1
        self.emit(indent + "}")
    
    fn generate_statement(inout self, node_ref: ASTNodeRef) -> String:
        """Generate MLIR for a statement.
        
        Args:
            node_ref: Reference to the statement node.
            
        Returns:
            Empty string (statements don't produce values).
        """
        let kind = self.parser.node_store.get_node_kind(node_ref)
        let indent = self.get_indent()
        
        if kind == ASTNodeKind.RETURN_STMT:
            # Get return node
            if node_ref < len(self.parser.return_nodes):
                let ret_node = self.parser.return_nodes[node_ref]
                if ret_node.value != 0:  # Has a return value
                    let value_ssa = self.generate_expression(ret_node.value)
                    let ret_type = self.get_expression_type(ret_node.value)
                    self.emit(indent + "return " + value_ssa + " : " + ret_type)
                else:
                    self.emit(indent + "return")
        elif kind == ASTNodeKind.VAR_DECL:
            # Variable declaration - generate as SSA value
            if node_ref < len(self.parser.var_decl_nodes):
                let var_node = self.parser.var_decl_nodes[node_ref]
                if var_node.initializer != 0:
                    let value_ssa = self.generate_expression(var_node.initializer)
                    # Track the identifier mapping
                    self.identifier_map[var_node.name] = value_ssa
        elif kind >= ASTNodeKind.BINARY_EXPR and kind <= ASTNodeKind.BOOL_LITERAL:
            # Expression statement (e.g., function call)
            _ = self.generate_expression(node_ref)
        
        return ""
    
    fn generate_expression(inout self, node_ref: ASTNodeRef) -> String:
        """Generate MLIR for an expression.
        
        Args:
            node_ref: Reference to the expression node.
            
        Returns:
            The MLIR value name (e.g., "%0", "%result").
        """
        let kind = self.parser.node_store.get_node_kind(node_ref)
        let indent = self.get_indent()
        
        if kind == ASTNodeKind.INTEGER_LITERAL:
            if node_ref < len(self.parser.int_literal_nodes):
                let lit_node = self.parser.int_literal_nodes[node_ref]
                let result = self.next_ssa_value()
                self.emit(indent + result + " = arith.constant " + lit_node.value + " : i64")
                return result
        
        elif kind == ASTNodeKind.STRING_LITERAL:
            if node_ref < len(self.parser.string_literal_nodes):
                let lit_node = self.parser.string_literal_nodes[node_ref]
                let result = self.next_ssa_value()
                self.emit(indent + result + ' = arith.constant "' + lit_node.value + '" : !mojo.string')
                return result
        
        elif kind == ASTNodeKind.FLOAT_LITERAL:
            if node_ref < len(self.parser.float_literal_nodes):
                let lit_node = self.parser.float_literal_nodes[node_ref]
                let result = self.next_ssa_value()
                self.emit(indent + result + " = arith.constant " + lit_node.value + " : f64")
                return result
        
        elif kind == ASTNodeKind.IDENTIFIER_EXPR:
            if node_ref < len(self.parser.identifier_nodes):
                let id_node = self.parser.identifier_nodes[node_ref]
                # Look up the identifier in the map
                if id_node.name in self.identifier_map:
                    return self.identifier_map[id_node.name]
                # If not found, return the name itself (could be a parameter)
                return id_node.name
        
        elif kind == ASTNodeKind.CALL_EXPR:
            return self.generate_call(node_ref)
        
        elif kind == ASTNodeKind.BINARY_EXPR:
            return self.generate_binary_expr(node_ref)
        
        return "%0"
    
    fn generate_call(inout self, node_ref: ASTNodeRef) -> String:
        """Generate function call.
        
        Args:
            node_ref: Reference to the call expression node.
            
        Returns:
            The result reference (or empty for void calls).
        """
        if node_ref >= len(self.parser.call_expr_nodes):
            return ""
        
        let call_node = self.parser.call_expr_nodes[node_ref]
        let indent = self.get_indent()
        
        # Check if it's a builtin
        if call_node.callee == "print":
            # Generate arguments
            var args = List[String]()
            for arg_ref in call_node.arguments:
                args.append(self.generate_expression(arg_ref[]))
            
            # Generate print call
            if len(args) > 0:
                let arg_type = self.get_expression_type(call_node.arguments[0])
                self.emit(indent + "mojo.print " + args[0] + " : " + arg_type)
            return ""
        else:
            # Regular function call
            var args = List[String]()
            var arg_types = List[String]()
            for arg_ref in call_node.arguments:
                args.append(self.generate_expression(arg_ref[]))
                arg_types.append(self.get_expression_type(arg_ref[]))
            
            let result = self.next_ssa_value()
            var call_str = indent + result + " = func.call @" + call_node.callee + "("
            for i in range(len(args)):
                if i > 0:
                    call_str += ", "
                call_str += args[i]
            call_str += ") : ("
            for i in range(len(arg_types)):
                if i > 0:
                    call_str += ", "
                call_str += arg_types[i]
            call_str += ") -> i64"  # Simplified - assume Int return
            self.emit(call_str)
            return result
    
    fn generate_binary_expr(inout self, node_ref: ASTNodeRef) -> String:
        """Generate binary operation.
        
        Args:
            node_ref: Reference to the binary expression node.
            
        Returns:
            The result reference.
        """
        if node_ref >= len(self.parser.binary_expr_nodes):
            return "%0"
        
        let bin_node = self.parser.binary_expr_nodes[node_ref]
        let indent = self.get_indent()
        
        # Generate left and right operands
        let left_val = self.generate_expression(bin_node.left)
        let right_val = self.generate_expression(bin_node.right)
        let result = self.next_ssa_value()
        
        # Determine the operation
        var op_name = ""
        if bin_node.operator == "+":
            op_name = "arith.addi"
        elif bin_node.operator == "-":
            op_name = "arith.subi"
        elif bin_node.operator == "*":
            op_name = "arith.muli"
        elif bin_node.operator == "/":
            op_name = "arith.divsi"
        elif bin_node.operator == "%":
            op_name = "arith.remsi"
        elif bin_node.operator == "==":
            op_name = "arith.cmpi eq,"
        elif bin_node.operator == "!=":
            op_name = "arith.cmpi ne,"
        elif bin_node.operator == "<":
            op_name = "arith.cmpi slt,"
        elif bin_node.operator == "<=":
            op_name = "arith.cmpi sle,"
        elif bin_node.operator == ">":
            op_name = "arith.cmpi sgt,"
        elif bin_node.operator == ">=":
            op_name = "arith.cmpi sge,"
        else:
            op_name = "arith.addi"  # Default
        
        # Get the type (assume i64 for now)
        let type_str = "i64"
        
        self.emit(indent + result + " = " + op_name + " " + left_val + ", " + right_val + " : " + type_str)
        return result
    
    fn get_expression_type(self, node_ref: ASTNodeRef) -> String:
        """Get the MLIR type of an expression.
        
        Args:
            node_ref: Reference to the expression node.
            
        Returns:
            The MLIR type string.
        """
        let kind = self.parser.node_store.get_node_kind(node_ref)
        
        if kind == ASTNodeKind.INTEGER_LITERAL:
            return "i64"
        elif kind == ASTNodeKind.STRING_LITERAL:
            return "!mojo.string"
        elif kind == ASTNodeKind.FLOAT_LITERAL:
            return "f64"
        elif kind == ASTNodeKind.BINARY_EXPR:
            return "i64"  # Simplified
        elif kind == ASTNodeKind.CALL_EXPR:
            return "i64"  # Simplified
        else:
            return "i64"  # Default
    
    fn emit(inout self, code: String):
        """Emit MLIR code to the output.
        
        Args:
            code: The MLIR code to emit.
        """
        self.output += code + "\n"
    
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
            return "!mojo.string"
        elif type_name == "NoneType" or type_name == "None":
            return "()"
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
        let indent = self.get_indent()
        if function_name == "print":
            # Generate print call
            if len(args) > 0:
                self.emit(indent + "mojo.print " + args[0])
            return ""
        else:
            # Generic builtin call
            var arg_str = ""
            for i in range(len(args)):
                if i > 0:
                    arg_str += ", "
                arg_str += args[i]
            self.emit(indent + "mojo.call_builtin @" + function_name + "(" + arg_str + ")")
            let result = self.next_ssa_value()
            return result
