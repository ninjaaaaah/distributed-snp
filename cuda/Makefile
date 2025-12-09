BUILD_DIR := build

.PHONY: all clean build run-snp-cpu run-snp-gpu run-snp-dist

all: build

build:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make

clean:
	rm -rf $(BUILD_DIR)

# Argument handling hack
# This allows passing arguments to make targets like: make run-snp-cpu 10
# It treats "10" as a target that does nothing.
ifeq (run-snp-cpu,$(firstword $(MAKECMDGOALS)))
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(RUN_ARGS):;@:)
endif

ifeq (run-snp-gpu,$(firstword $(MAKECMDGOALS)))
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(RUN_ARGS):;@:)
endif

ifeq (run-snp-dist,$(firstword $(MAKECMDGOALS)))
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(RUN_ARGS):;@:)
endif

run-snp-cpu: build
	@echo "Running CPU Benchmark with N=$(RUN_ARGS)..."
	@./$(BUILD_DIR)/snp_benchmark_cpu $(RUN_ARGS)

run-snp-gpu: build
	@echo "Running GPU Benchmark with N=$(RUN_ARGS)..."
	@./$(BUILD_DIR)/snp_benchmark_gpu $(RUN_ARGS)

run-snp-dist: build
	@echo "Running Distributed Benchmark with N=$(RUN_ARGS)..."
	@./$(BUILD_DIR)/snp_benchmark_dist $(RUN_ARGS)
