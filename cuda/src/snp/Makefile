# Compiler settings
CXX      = g++
CXXFLAGS = -Wall -g

# Directories and Target Names
BUILD_DIR = ./build
TARGET    = $(BUILD_DIR)/snp_system
SRC       = snp.cpp

# Default rule: Build the executable
all: $(TARGET)

# Rule to link the executable (creates build dir if missing)
$(TARGET): $(SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Rule to run the program
run: $(TARGET)
	@echo "Running simulation..."
	@./$(TARGET)

# Rule to clean up (removes the build directory)
clean:
	rm -rf $(BUILD_DIR)

# Phony targets (not actual files)
.PHONY: all run clean