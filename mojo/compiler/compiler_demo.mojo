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

"""Demo of the open source Mojo compiler.

This demonstrates the basic structure and capabilities of the compiler
as outlined in the open-source-compiler.md proposal.
"""

from src import CompilerOptions, compile


fn main() raises:
    """Demonstrate the compiler with a simple example."""
    print("=== Mojo Open Source Compiler Demo ===")
    print()
    print("This compiler implements the design from:")
    print("  mojo/proposals/open-source-compiler.md")
    print()
    print("Architecture:")
    print("  1. Frontend: Lexer and Parser")
    print("  2. Semantic Analysis: Type Checker")
    print("  3. IR Generation: MLIR Generator")
    print("  4. Optimization: MLIR Optimizer")
    print("  5. Code Generation: LLVM Backend")
    print()
    
    # Create compiler options
    var options = CompilerOptions(
        target="x86_64-linux",
        opt_level=2,
        stdlib_path="../stdlib",
        debug=False,
        output_path="hello_world"
    )
    
    print("Compiler Options:")
    print("  Target:", options.target)
    print("  Optimization Level:", options.opt_level)
    print("  Standard Library:", options.stdlib_path)
    print("  Debug:", options.debug)
    print("  Output:", options.output_path)
    print()
    
    # Example: Compile a simple program
    print("Example compilation workflow:")
    print()
    
    let example_source = """
fn main():
    print("Hello, World!")
"""
    
    print("Source code:")
    print(example_source)
    print()
    
    print("Compilation phases:")
    print("  [1] Lexing: Tokenizing source code...")
    print("  [2] Parsing: Building Abstract Syntax Tree...")
    print("  [3] Type Checking: Validating types and semantics...")
    print("  [4] IR Generation: Lowering to MLIR...")
    print("  [5] Optimization: Applying optimization passes...")
    print("  [6] Code Generation: Generating native code...")
    print()
    
    print("Phase 1 Status: Minimal Viable Compiler")
    print("  [~] Lexer for basic syntax")
    print("  [~] Parser for functions and expressions")
    print("  [ ] Type checker for simple types")
    print("  [ ] MLIR code generation")
    print("  [ ] LLVM backend integration")
    print()
    
    print("Next steps:")
    print("  - Complete lexer implementation")
    print("  - Implement parser for function definitions")
    print("  - Add basic type checking")
    print("  - Integrate MLIR/LLVM infrastructure")
    print("  - Create end-to-end compilation test")
    print()
    
    print("For more information, see:")
    print("  - mojo/compiler/README.md")
    print("  - mojo/proposals/open-source-compiler.md")
