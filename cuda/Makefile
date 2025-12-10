BUILD_DIR := build
DIST_BINARY := $(BUILD_DIR)/snp_benchmark_dist

# MPI Configuration
NODES ?= localhost,10.0.0.2
HOSTFILE ?= hostfile
NP ?= 2
REMOTE_DIR ?= /home/shared/tmp/distributed-snp

.PHONY: all clean build run-snp-cpu run-snp-gpu run-snp-dist distribute run-mpi

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

# Distribute binary to remote nodes
distribute: build
	@echo "Distributing binaries to nodes: $(NODES)..."
	@for node in $(shell echo $(NODES) | tr ',' ' '); do \
		echo "Copying to $$node:$(REMOTE_DIR)..."; \
		ssh $$node "mkdir -p $(REMOTE_DIR)"; \
		scp $(DIST_BINARY) $$node:$(REMOTE_DIR)/; \
	done
	@echo "Distribution complete!"

# Create hostfile for MPI
hostfile:
	@echo "Creating hostfile..."
	@> $(HOSTFILE)
	@for node in $(shell echo $(NODES) | tr ',' ' '); do \
		echo "$$node" >> $(HOSTFILE); \
	done
	@echo "Hostfile created with nodes: $(NODES)"

# Run MPI program with hostfile
run-snp-dist: build distribute hostfile
	@echo "Running MPI program on $(NP) processes across nodes..."
	@mpirun -np $(NP) \
		--mca btl_tcp_if_include ens5 \
		--mca oob_tcp_if_include ens5 \
		--hostfile $(HOSTFILE) \
		$(REMOTE_DIR)/snp_benchmark_dist $(RUN_ARGS)

# Run MPI program locally (no distribution needed)
run-snp-dist-local: build
	@echo "Running MPI program locally on $(NP) processes with N=$(RUN_ARGS)..."
	@mpirun -np $(NP) ./$(BUILD_DIR)/snp_benchmark_dist $(RUN_ARGS)
