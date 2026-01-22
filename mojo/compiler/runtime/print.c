/*===----------------------------------------------------------------------===*
 * Copyright (c) 2025, Modular Inc. All rights reserved.
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions:
 * https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *===----------------------------------------------------------------------===*/

/**
 * Mojo Runtime Library - Print Functions
 * 
 * This module provides basic print functionality for the Mojo compiler.
 * It implements runtime support for the mojo.print operation.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

/**
 * Print a null-terminated string to stdout
 */
void _mojo_print_string(const char* str) {
    printf("%s\n", str);
    fflush(stdout);
}

/**
 * Print a 64-bit signed integer to stdout
 */
void _mojo_print_int(int64_t value) {
    printf("%lld\n", (long long)value);
    fflush(stdout);
}

/**
 * Print a 64-bit floating point number to stdout
 */
void _mojo_print_float(double value) {
    printf("%f\n", value);
    fflush(stdout);
}

/**
 * Print a boolean value to stdout (True/False)
 */
void _mojo_print_bool(bool value) {
    printf("%s\n", value ? "True" : "False");
    fflush(stdout);
}
