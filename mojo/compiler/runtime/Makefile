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

# Mojo Runtime Library Build System

CC = cc
AR = ar
CFLAGS = -O2 -fPIC -Wall -Wextra
ARFLAGS = rcs

# Output library
LIB = libmojo_runtime.a

# Source files
SOURCES = print.c
OBJECTS = $(SOURCES:.c=.o)

# Default target
all: $(LIB)

# Build the static library
$(LIB): $(OBJECTS)
	@echo "Creating static library: $(LIB)"
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS)

# Compile C source files
%.o: %.c
	@echo "Compiling: $<"
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts"
	rm -f $(OBJECTS) $(LIB)

# Phony targets
.PHONY: all clean
