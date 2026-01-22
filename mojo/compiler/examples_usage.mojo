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

"""Example demonstrating how to use the Mojo compiler programmatically.

This shows the intended API for using the compiler as a library,
compiling Mojo source code to executables.
"""

from src import CompilerOptions, compile


fn compile_hello_world() raises:
    """Compile the Hello World example."""
    print("=" * 70)
    print("Example 1: Compile Hello World")
    print("=" * 70)
    
    # Configure compiler options
    var options = CompilerOptions(
        target="x86_64-linux",
        opt_level=2,
        stdlib_path="../stdlib",
        debug=False,
        output_path="hello_world"
    )
    
    print("\nCompiling examples/hello_world.mojo...")
    print("Target:", options.target)
    print("Optimization level:", options.opt_level)
    print()
    
    # Note: This will work once all components are complete
    # let success = compile("examples/hello_world.mojo", options)
    # if success:
    #     print("\n✓ Compilation successful!")
    #     print("Run with: ./hello_world")
    
    print("Note: Full compilation will work once all components are complete")


fn compile_with_debug() raises:
    """Compile with debug information."""
    print("\n" + "=" * 70)
    print("Example 2: Compile with Debug Info")
    print("=" * 70)
    
    var options = CompilerOptions(
        target="native",  # Use native architecture
        opt_level=0,      # No optimization for debugging
        stdlib_path="../stdlib",
        debug=True,       # Include debug info
        output_path="hello_world_debug"
    )
    
    print("\nCompiling with debug information...")
    print("Debug mode:", options.debug)
    print("Optimization:", options.opt_level, "(disabled for debugging)")
    print()
    
    print("When complete, this will:")
    print("  - Include DWARF debug information")
    print("  - Preserve source line mappings")
    print("  - Enable debugging with gdb/lldb")


fn compile_optimized() raises:
    """Compile with aggressive optimization."""
    print("\n" + "=" * 70)
    print("Example 3: Aggressive Optimization")
    print("=" * 70)
    
    var options = CompilerOptions(
        target="x86_64-linux",
        opt_level=3,      # Maximum optimization
        stdlib_path="../stdlib",
        debug=False,
        output_path="program_optimized"
    )
    
    print("\nCompiling with aggressive optimization...")
    print("Optimization level:", options.opt_level)
    print()
    
    print("Optimization passes enabled:")
    print("  - Inlining (aggressive)")
    print("  - Constant folding")
    print("  - Dead code elimination")
    print("  - Loop optimizations")
    print("  - Move/copy elimination")
    print("  - Trait devirtualization")


fn demonstrate_compilation_pipeline() raises:
    """Show the compilation pipeline in detail."""
    print("\n" + "=" * 70)
    print("Example 4: Compilation Pipeline")
    print("=" * 70)
    
    let source_code = """
fn add(a: Int, b: Int) -> Int:
    return a + b

fn main():
    let result = add(40, 2)
    print(result)
"""
    
    print("\nSource code:")
    print(source_code)
    
    print("\nCompilation pipeline:")
    print()
    print("1. [Lexer] Tokenize source code")
    print("   → Produces tokens: 'fn', 'add', '(', 'a', ':', 'Int', ...")
    print()
    print("2. [Parser] Build Abstract Syntax Tree (AST)")
    print("   → Produces AST with FunctionNode, BinaryExprNode, etc.")
    print()
    print("3. [Type Checker] Validate types")
    print("   → Check: add(Int, Int) -> Int is valid")
    print("   → Check: print(Int) is valid")
    print()
    print("4. [MLIR Generator] Lower to MLIR")
    print("   → func.func @add(%arg0: i64, %arg1: i64) -> i64 {")
    print("   →   %0 = arith.addi %arg0, %arg1 : i64")
    print("   →   return %0 : i64")
    print("   → }")
    print()
    print("5. [Optimizer] Apply optimization passes")
    print("   → Inline small functions")
    print("   → Fold constants")
    print("   → Eliminate dead code")
    print()
    print("6. [LLVM Backend] Generate machine code")
    print("   → Lower MLIR to LLVM IR")
    print("   → Generate object file (.o)")
    print("   → Link with runtime and stdlib")
    print()
    print("7. [Output] Native executable")
    print("   → ./program")


fn show_api_usage() raises:
    """Show how to use the compiler API."""
    print("\n" + "=" * 70)
    print("Example 5: Compiler API Usage")
    print("=" * 70)
    
    print("\nUsing the compiler as a library:")
    print()
    
    print("```mojo")
    print("from compiler import CompilerOptions, compile")
    print()
    print("fn main() raises:")
    print("    var options = CompilerOptions(")
    print("        target=\"x86_64-linux\",")
    print("        opt_level=2,")
    print("        stdlib_path=\"/path/to/stdlib\"")
    print("    )")
    print("    ")
    print("    let success = compile(\"myprogram.mojo\", options)")
    print("    if success:")
    print("        print(\"Compilation successful!\")")
    print("```")
    print()
    
    print("Using the compiler CLI (future):")
    print()
    print("```bash")
    print("# Basic compilation")
    print("mojo-compiler myprogram.mojo")
    print()
    print("# With options")
    print("mojo-compiler --target x86_64-linux \\")
    print("              --opt-level 3 \\")
    print("              --output myprogram \\")
    print("              myprogram.mojo")
    print()
    print("# Run tests")
    print("mojo-compiler test ./test/")
    print()
    print("# Build package")
    print("mojo-compiler build --manifest mojo.toml")
    print("```")


fn main() raises:
    """Run all examples."""
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "Mojo Open Source Compiler Examples" + " " * 19 + "║")
    print("╚" + "=" * 68 + "╝")
    
    print("\nThis demonstrates how to use the Mojo compiler programmatically")
    print("and shows what the compilation pipeline looks like internally.")
    
    # Run all examples
    compile_hello_world()
    compile_with_debug()
    compile_optimized()
    demonstrate_compilation_pipeline()
    show_api_usage()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("The Mojo compiler provides:")
    print("  ✓ Flexible compilation API")
    print("  ✓ Multiple optimization levels")
    print("  ✓ Debug information support")
    print("  ✓ Cross-platform target support")
    print("  ✓ Integration with existing toolchains")
    print()
    print("For more information:")
    print("  - README.md - Overview and status")
    print("  - IMPLEMENTATION_PROGRESS.md - Latest updates")
    print("  - DEVELOPER_GUIDE.md - How to contribute")
    print("  - ../proposals/open-source-compiler.md - Full specification")
    print()
    print("Current Status: Phase 1 at 55% completion")
    print("Next Milestone: Complete parser and type checker")
    print()
