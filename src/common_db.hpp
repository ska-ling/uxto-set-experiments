#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <filesystem>
#include <numeric>
#include <cmath>
#include <string>
#include <unordered_map>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <regex>
#include <fstream>
#include <set>


#define FMT_HEADER_ONLY 1
#include <fmt/core.h>

#if defined(KTH_LEVELDB)
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#elif defined(KTH_LMDB)
#include <lmdb.h>
#elif defined(KTH_MEMORY)
#include <boost/unordered/unordered_flat_map.hpp>
#endif
using MapStats = std::unordered_map<std::string, double>;

struct BenchResult {
    int64_t time_ns;
    size_t num_entries;
};

struct BenchStats {
    std::vector<BenchResult> insertion_results;
    std::vector<BenchResult> deletion_results;
    
#if defined(KTH_FLOWEE)    
    std::vector<BenchResult> block_commit_results;
#endif

    std::vector<MapStats> db_stats;
    std::vector<MapStats> so_stats;
    std::vector<int64_t> sync_durations_ns;
};

constexpr size_t UTXO_KEY_SIZE = 36;   // 32 bytes hash + 4 bytes index
constexpr size_t UTXO_MIN_VALUE = 30;
constexpr size_t UTXO_MAX_VALUE = 38;
// constexpr size_t NUM_ENTRIES = 100000;
// constexpr double DELETE_RATIO = 0.3;


using utxo_key_t = std::array<uint8_t, UTXO_KEY_SIZE>;
using utxo_value_t = std::vector<uint8_t>;
using utxo_value_static_t = std::pair<uint32_t, std::array<uint8_t, UTXO_MAX_VALUE>>;
// blockHeigt, blockOffset
using utxo_value_simple_t = std::pair<int, int>;

struct utxo_entry {
    utxo_key_t key;
    utxo_value_t value;
};

struct utxo_entry_simple {
    utxo_key_t key;
    utxo_value_simple_t value;
};

struct utxo_entry_static {
    utxo_key_t key;
    utxo_value_static_t value;
};
