#!/bin/bash

# Script para compilar y ejecutar el benchmark de lookup

set -e

echo "=== Building UTXO Lookup Benchmark ==="

# Build the project
./build.sh

echo ""
echo "=== Running Lookup Benchmark ==="
echo "Note: Make sure the UTXO database is pre-synced to block 750,000"
echo ""

# Run the lookup benchmark
cd build/build/Release/bin
./bench_lookup

echo ""
echo "=== Benchmark completed ==="
