# Compiler settings
CXX      = g++
CXXFLAGS = -Wall -g

# Directories and Target Names
BUILD_DIR = ./build
TARGET    = $(BUILD_DIR)/snp_system
SRCS      = snp.cpp snp_system.cpp linear_algebra.cpp sorting.cpp
OBJS      = $(BUILD_DIR)/snp.o $(BUILD_DIR)/snp_system.o $(BUILD_DIR)/linear_algebra.o $(BUILD_DIR)/sorting.o

# Default rule: Build the executable
all: $(TARGET)

# Rule to link the executable (creates build dir if missing)
$(TARGET): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Rule to compile individual source files
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to run the program
run: $(TARGET)
	@echo "Running simulation..."
	@./$(TARGET)

# Rule to clean up (removes the build directory)
clean:
	rm -rf $(BUILD_DIR)

# Phony targets (not actual files)
.PHONY: all run clean