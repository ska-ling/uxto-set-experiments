#pragma once

#include <boost/interprocess/managed_mapped_file.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <boost/container_hash/hash.hpp>

#include <filesystem>
#include <vector>
#include <array>
#include <span>
#include <optional>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <variant>
#include <fmt/format.h>
#include <fmt/core.h>

#include "common_db.hpp"
#include "log.hpp"

namespace utxo {

namespace bip = boost::interprocess;
namespace fs = std::filesystem;

// Helper for compile-time operations
template <size_t N, typename Func, size_t... Is>
constexpr void for_each_index_impl(Func&& f, std::index_sequence<Is...>) {
    (f(std::integral_constant<size_t, Is>{}), ...);
}

template <size_t N, typename Func>
constexpr void for_each_index(Func&& f) {
    for_each_index_impl<N>(std::forward<Func>(f), std::make_index_sequence<N>{});
}

constexpr size_t operator"" _mib(unsigned long long n) {
    return static_cast<size_t>(n) * 1024 * 1024;
}

// Constants
inline constexpr size_t utxo_key_size = 36;
inline constexpr std::array<size_t, 4> container_sizes = {44, 128, 512, 10240};
inline constexpr std::array<size_t, 4> file_sizes = {
    400_mib,
    400_mib,
    200_mib,
    100_mib
};
inline constexpr size_t index_file_size = 400_mib; // Size for index files

// Index pointer structure (64 bits total)
// - 2 bits: container index (0-3)
// - 16 bits: version
// - 46 bits: entry index within file
struct index_pointer {
    uint64_t value;
    
    static constexpr uint64_t CONTAINER_BITS = 2;
    static constexpr uint64_t VERSION_BITS = 16;
    static constexpr uint64_t INDEX_BITS = 46;
    
    static constexpr uint64_t CONTAINER_MASK = (1ULL << CONTAINER_BITS) - 1;
    static constexpr uint64_t VERSION_MASK = (1ULL << VERSION_BITS) - 1;
    static constexpr uint64_t INDEX_MASK = (1ULL << INDEX_BITS) - 1;
    
    index_pointer() : value(0) {}
    index_pointer(uint64_t container, uint64_t version, uint64_t index) {
        value = (container & CONTAINER_MASK) << (VERSION_BITS + INDEX_BITS) |
                (version & VERSION_MASK) << INDEX_BITS |
                (index & INDEX_MASK);
    }
    
    uint64_t container() const { return (value >> (VERSION_BITS + INDEX_BITS)) & CONTAINER_MASK; }
    uint64_t version() const { return (value >> INDEX_BITS) & VERSION_MASK; }
    uint64_t index() const { return value & INDEX_MASK; }
};

// Types
using utxo_key_t = std::array<uint8_t, utxo_key_size>;
using data_key_t = uint64_t; // Key for data maps (was utxo_key_t, now index)

// Helper to create variant for compile-time dispatch
auto make_index_variant(size_t index) {
    using variant_t = std::variant<
        std::integral_constant<size_t, 0>,
        std::integral_constant<size_t, 1>,
        std::integral_constant<size_t, 2>,
        std::integral_constant<size_t, 3>
    >;
    
    switch (index) {
        case 0: return variant_t{std::integral_constant<size_t, 0>{}};
        case 1: return variant_t{std::integral_constant<size_t, 1>{}};
        case 2: return variant_t{std::integral_constant<size_t, 2>{}};
        case 3: return variant_t{std::integral_constant<size_t, 3>{}};
        default: throw std::out_of_range("Invalid index");
    }
}

void print_key(utxo_key_t const& key) {
    // first 32 bytes are the transaction hash, print in hex reversed
    for (size_t i = 0; i < 32; ++i) {
        log_print("{:02x}", key[31 - i]);
    }   
    // the last 4 bytes are the output index, print as integer
    uint32_t output_index = 0;
    std::copy(key.end() - 4, key.end(), reinterpret_cast<uint8_t*>(&output_index));
    log_print(":{}", output_index);
    log_print("\n");
}

using segment_manager_t = bip::managed_mapped_file::segment_manager;
using key_hash = boost::hash<utxo_key_t>;
using key_equal = std::equal_to<utxo_key_t>;

// Index map type
using index_map_t = boost::unordered_flat_map<
    utxo_key_t,
    index_pointer,
    key_hash,
    key_equal,
    bip::allocator<std::pair<utxo_key_t const, index_pointer>, segment_manager_t>
>;

// Select appropriate uint type for size
template <size_t Size>
using size_type = std::conditional_t<Size <= 255, uint8_t, uint16_t>;

// Simplified value structure
template <size_t Size>
struct utxo_value {
    uint32_t block_height;
    size_type<Size> actual_size;
    std::array<uint8_t, Size - sizeof(uint32_t) - sizeof(size_type<Size>)> data;
    
    void set_data(std::span<uint8_t const> output_data) {
        actual_size = static_cast<size_type<Size>>(output_data.size());
        auto const copy_size = std::min(output_data.size(), data.size());
        std::ranges::copy(output_data.first(copy_size), data.begin());
    }
    
    std::span<uint8_t const> get_data() const {
        return {data.data(), actual_size};
    }
};

// Data map type (now uses data_key_t instead of utxo_key_t)
template <size_t Size>
using utxo_map = boost::unordered_flat_map<
    data_key_t,
    utxo_value<Size>,
    boost::hash<data_key_t>,
    std::equal_to<data_key_t>,
    bip::allocator<std::pair<data_key_t const, utxo_value<Size>>, segment_manager_t>
>;

// OP_RETURN set type
using op_return_set_t = boost::unordered_flat_set<
    utxo_key_t,
    key_hash,
    key_equal,
    bip::allocator<utxo_key_t, segment_manager_t>
>;

// File metadata
struct file_metadata {
    uint32_t min_block_height = UINT32_MAX;
    uint32_t max_block_height = 0;
    size_t entry_count = 0;
    size_t container_index = 0;
    size_t version = 0;
    uint64_t next_index = 0; // Next available index for this file
    
    file_metadata() = default;
    
    bool block_in_range(uint32_t height) const {
        return entry_count > 0 && 
            height >= min_block_height && height <= max_block_height;
    }
    
    void update_on_insert(uint32_t height) {
        if (entry_count == 0) {
            min_block_height = max_block_height = height;
        } else {
            min_block_height = std::min(min_block_height, height);
            max_block_height = std::max(max_block_height, height);
        }
        entry_count++;
    }
    
    void update_on_delete() {
        if (entry_count > 0) --entry_count;
    }
    
    uint64_t allocate_index() {
        return next_index++;
    }
};

// Index metadata
struct index_metadata {
    size_t entry_count = 0;
    size_t version = 0;
};

// Search record
struct search_record {
    uint32_t access_height;
    uint32_t insertion_height;
    uint32_t depth;
    bool is_cache_hit;
    bool found;
    char operation;
    
    uint32_t get_utxo_age() const {
        return found && 
            access_height >= insertion_height ? access_height - insertion_height : 0;
    }
};

// Search statistics
class search_stats {
public:
    void add_record(uint32_t access_h, uint32_t insert_h, uint32_t d, bool cache, bool f, char op) {
        search_records.emplace_back(search_record{access_h, insert_h, d, cache, f, op});
    }
    
    void reset() { 
        search_records.clear(); 
    }
    
    struct summary {
        size_t total_operations = 0;
        size_t found_operations = 0;
        size_t current_version_hits = 0;
        size_t cache_hits = 0;
        double avg_depth = 0.0;
        double avg_utxo_age = 0.0;
        double cache_hit_rate = 0.0;
        double hit_rate = 0.0;
    };
    
    summary get_summary() const {
        summary s;
        if (search_records.empty()) return s;
        
        size_t total_depth = 0;
        size_t total_age = 0;
        size_t cache_accesses = 0;
        
        for (auto const& record : search_records) {
            s.total_operations++;
            total_depth += record.depth;
            
            if (record.found) {
                s.found_operations++;
                if (record.depth == 0) s.current_version_hits++;
                total_age += record.get_utxo_age();
            }
            
            if (record.depth > 0) {
                cache_accesses++;
                if (record.is_cache_hit) s.cache_hits++;
            }
        }
        
        if (s.total_operations > 0) {
            s.avg_depth = double(total_depth) / double(s.total_operations);
            s.hit_rate = double(s.found_operations) / double(s.total_operations);
        }
        if (s.found_operations > 0) {
            s.avg_utxo_age = double(total_age) / double(s.found_operations);
        }
        if (cache_accesses > 0) {
            s.cache_hit_rate = double(s.cache_hits) / double(cache_accesses);
        }
        
        return s;
    }
    
private:
    std::vector<search_record> search_records;
};

// File cache for both index and data files
struct file_cache {
    template <typename MapType>
    using ret_t = std::pair<MapType&, bool>;
    
    using file_key_t = std::pair<size_t, size_t>; // (container_index, version)
    using index_file_key_t = size_t; // Just version for index files

    explicit 
    file_cache(std::string const& path, size_t max_size = 1) 
        : base_path_(path), 
          max_cached_files_(max_size)
    {}
    
    // Get or open data file
    template <size_t Index>
    ret_t<utxo_map<container_sizes[Index]>> get_or_open_data_file(size_t container_idx, size_t version) {
        file_key_t file_key{container_idx, version};
        
        ++gets_;
        auto const now = std::chrono::steady_clock::now();
        
        // Track access
        ++access_frequency_[file_key];
        
        // Check cache
        if (auto it = data_cache_.find(file_key); it != data_cache_.end()) {
            it->second.last_used = now;
            ++it->second.access_count;
            ++hits_;
            return {*static_cast<utxo_map<container_sizes[Index]>*>(it->second.map_ptr), true};
        }
        
        // Evict if needed
        if (data_cache_.size() + index_cache_.size() >= max_cached_files_) {
            evict_lru();
        }
        
        // Open file
        try {
            auto file_path = make_data_file_path(container_idx, version);
            auto segment = std::make_unique<bip::managed_mapped_file>(
                bip::open_only, file_path.c_str());
            
            auto* map = segment->find<utxo_map<container_sizes[Index]>>("db_map").first;
            if (!map) {
                throw std::runtime_error("Map not found in file");
            }
            
            data_cache_[file_key] = {
                .segment = std::move(segment),
                .map_ptr = map,
                .last_used = now,
                .access_count = 1,
                .is_pinned = false
            };
            
            return {*map, false};
        } catch (std::exception const& e) {
            auto file_path = make_data_file_path(container_idx, version);
            log_print("file_cache: ERROR opening data file {}: {}\n", file_path, e.what());
            throw;
        }
    }
    
    // Get or open index file
    ret_t<index_map_t> get_or_open_index_file(size_t version) {
        ++gets_;
        auto const now = std::chrono::steady_clock::now();
        
        // Check cache
        if (auto it = index_cache_.find(version); it != index_cache_.end()) {
            it->second.last_used = now;
            ++it->second.access_count;
            ++hits_;
            return {*static_cast<index_map_t*>(it->second.map_ptr), true};
        }
        
        // Evict if needed
        if (data_cache_.size() + index_cache_.size() >= max_cached_files_) {
            evict_lru();
        }
        
        // Open file
        try {
            auto file_path = make_index_file_path(version);
            auto segment = std::make_unique<bip::managed_mapped_file>(
                bip::open_only, file_path.c_str());
            
            auto* map = segment->find<index_map_t>("index_map").first;
            if (!map) {
                throw std::runtime_error("Index map not found in file");
            }
            
            index_cache_[version] = {
                .segment = std::move(segment),
                .map_ptr = map,
                .last_used = now,
                .access_count = 1,
                .is_pinned = false
            };
            
            return {*map, false};
        } catch (std::exception const& e) {
            auto file_path = make_index_file_path(version);
            log_print("file_cache: ERROR opening index file {}: {}\n", file_path, e.what());
            throw;
        }
    }
    
    float get_hit_rate() const {
        return gets_ > 0 ? float(hits_) / float(gets_) : 0.0f;
    }
    
    void set_cache_size(size_t new_size) { 
        max_cached_files_ = new_size; 
    }
    
private:
    struct cached_file {
        std::unique_ptr<bip::managed_mapped_file> segment;
        void* map_ptr;
        std::chrono::steady_clock::time_point last_used;
        size_t access_count = 0;
        bool is_pinned = false;
    };
    
    std::string make_data_file_path(size_t index, size_t version) const {
        return fmt::format("{}/cont_{}_v{:05}.dat", base_path_, index, version);
    }
    
    std::string make_index_file_path(size_t version) const {
        return fmt::format("{}/index_v{:05}.dat", base_path_, version);
    }
    
    void evict_lru() {
        // Find LRU from both caches
        std::chrono::steady_clock::time_point oldest_time = std::chrono::steady_clock::now();
        bool evict_from_data = true;
        file_key_t data_key_to_evict;
        index_file_key_t index_key_to_evict;
        
        // Check data cache
        for (auto const& [key, file] : data_cache_) {
            if (!file.is_pinned && file.last_used < oldest_time) {
                oldest_time = file.last_used;
                data_key_to_evict = key;
                evict_from_data = true;
            }
        }
        
        // Check index cache
        for (auto const& [key, file] : index_cache_) {
            if (!file.is_pinned && file.last_used < oldest_time) {
                oldest_time = file.last_used;
                index_key_to_evict = key;
                evict_from_data = false;
            }
        }
        
        // Evict
        if (evict_from_data && !data_cache_.empty()) {
            data_cache_.erase(data_key_to_evict);
        } else if (!index_cache_.empty()) {
            index_cache_.erase(index_key_to_evict);
        }
        ++evictions_;
    }
    
    boost::unordered_flat_map<file_key_t, cached_file> data_cache_;
    boost::unordered_flat_map<index_file_key_t, cached_file> index_cache_;
    boost::unordered_flat_map<file_key_t, size_t> access_frequency_;
    size_t max_cached_files_;
    size_t gets_ = 0;
    size_t hits_ = 0;
    size_t evictions_ = 0;
    std::string base_path_;
};

// Container statistics
struct container_stats {
    size_t total_inserts = 0;
    size_t total_deletes = 0;
    size_t current_size = 0;
    size_t failed_deletes = 0;
    size_t rehash_count = 0;
    std::map<size_t, size_t> value_size_distribution;
};

// OP_RETURN statistics
struct op_return_stats {
    size_t total_inserts = 0;
    size_t total_deletes = 0;
    size_t current_size = 0;
    size_t failed_deletes = 0;
};

// Main database class
class utxo_db {
    using span_bytes = std::span<uint8_t const>;
    static constexpr auto IdxN = container_sizes.size();
    static constexpr std::string_view op_return_file_name = "op_return_set.dat";

public:
    void configure(std::string_view path, bool remove_existing = false) {
        db_path_ = path;
        
        if (remove_existing && fs::exists(path)) {
            fs::remove_all(path);
        }
        fs::create_directories(path);
        
        // Configure file cache
        file_cache_ = file_cache(std::string(path), 10); // Increase cache size
        
        // Initialize OP_RETURN set
        open_or_create_op_return_set();

        // Find optimal buckets
        static_assert(IdxN == 4);
        min_buckets_ok_[0] = find_optimal_buckets<0>("./optimal", file_sizes[0], 7864304);
        log_print("Optimal number of buckets for container {} and file size {}: {}\n", 0, file_sizes[0], min_buckets_ok_[0]);
        min_buckets_ok_[1] = find_optimal_buckets<1>("./optimal", file_sizes[1], 7864304);
        log_print("Optimal number of buckets for container {} and file size {}: {}\n", 1, file_sizes[1], min_buckets_ok_[1]);
        min_buckets_ok_[2] = find_optimal_buckets<2>("./optimal", file_sizes[2], 7864304);
        log_print("Optimal number of buckets for container {} and file size {}: {}\n", 2, file_sizes[2], min_buckets_ok_[2]);
        min_buckets_ok_[3] = find_optimal_buckets<3>("./optimal", file_sizes[3], 7864304);
        log_print("Optimal number of buckets for container {} and file size {}: {}\n", 3, file_sizes[3], min_buckets_ok_[3]);

        // Find optimal buckets for index
        index_min_buckets_ok_ = find_optimal_index_buckets("./optimal", index_file_size, 15728608);
        log_print("Optimal number of buckets for index and file size {}: {}\n", index_file_size, index_min_buckets_ok_);

        // Initialize index
        size_t latest_index_version = find_latest_index_version_from_files();
        open_or_create_index(latest_index_version);
        
        // Load index metadata
        for (size_t v = 0; v <= latest_index_version; ++v) {
            load_index_metadata_from_disk(v);
        }

        // Initialize containers
        for_each_index<IdxN>([&](auto I) {
            size_t latest_version = find_latest_version_from_files(I);
            open_or_create_container<I>(latest_version);
            
            // Load metadata
            for (size_t v = 0; v <= latest_version; ++v) {
                load_metadata_from_disk(I, v);
            }
        });
    }
    
    void close() {
        for_each_index<IdxN>([&](auto I) {
            close_container<I>();
        });
        close_index();
        close_op_return_set();
    }
    
    size_t size() const {
        return entries_count_;
    }

    // Insert
    bool insert(utxo_key_t const& key, span_bytes value, uint32_t height) {
        size_t const container_idx = get_index_from_size(value.size());
        if (container_idx >= IdxN) {
            log_print("insert: Invalid index {} for value size {}. Height: {}\n", 
                     container_idx, value.size(), height);
            print_key(key);
            throw std::out_of_range("Value size too large");
        }
        
        return std::visit([&](auto ic) {
            return insert_in_container<ic>(key, value, height);
        }, make_index_variant(container_idx));
    }

    // Insert OP_RETURNs
    void insert_op_returns(boost::unordered_flat_set<utxo_key_t> const& op_return_keys, uint32_t height) {
        if (!op_return_segment_ || !op_return_set_) {
            log_print("ERROR: OP_RETURN set not initialized before insert_op_returns.\n");
            return;
        }

        for (auto const& key : op_return_keys) {
            auto [it, success] = op_return_set_->insert(key);
            if (success) {
                ++op_return_stats_.total_inserts;
                ++op_return_stats_.current_size;
            }
        }
    }
    
    // Erase
    size_t erase(utxo_key_t const& key, uint32_t height) {
        // First try to find in index
        auto pointer_opt = find_in_index(key, height, 'e');
        
        if (!pointer_opt) {
            // Check OP_RETURN set
            if (op_return_set_ && op_return_set_->count(key)) {
                if (op_return_set_->erase(key)) {
                    ++op_return_stats_.total_deletes;
                    --op_return_stats_.current_size;
                    return 1;
                }
            }
            return 0;
        }
        
        auto ptr = pointer_opt.value();
        
        // Erase from data file
        bool erased = std::visit([&](auto ic) {
            return erase_from_data_file<ic>(ptr, height);
        }, make_index_variant(ptr.container()));
        
        if (erased) {
            // Erase from index
            erase_from_index(key);
            entries_count_--;
            return 1;
        }
        
        log_print("ERROR: Index pointed to non-existent data! DB might be corrupted.\n");
        return 0;
    }
    
    // Find
    std::optional<std::vector<uint8_t>> find(utxo_key_t const& key, uint32_t height) {
        // First try to find in index
        auto pointer_opt = find_in_index(key, height, 'f');
        
        if (!pointer_opt) {
            return std::nullopt;
        }
        
        auto ptr = pointer_opt.value();
        
        // Find in data file
        return std::visit([&](auto ic) {
            return find_in_data_file<ic>(ptr, height);
        }, make_index_variant(ptr.container()));
    }
    
    // Stats
    search_stats const& get_search_stats() const { 
        return search_stats_; 
    }

    void reset_search_stats() { 
        search_stats_.reset(); 
    }
    
    float get_cache_hit_rate() const {
        return file_cache_.get_hit_rate();
    }
    
    struct db_statistics {
        size_t total_entries;
        size_t total_inserts;
        size_t total_deletes;
        std::array<container_stats, IdxN> containers;
        float cache_hit_rate;
        search_stats::summary search_summary;
        std::array<size_t, IdxN> rotations_per_container;
        size_t index_rotations;
        op_return_stats op_return;
    };
    
    db_statistics get_statistics() {
        db_statistics stats;
        
        stats.total_entries = entries_count_;
        stats.cache_hit_rate = get_cache_hit_rate();
        stats.search_summary = search_stats_.get_summary();
        stats.index_rotations = current_index_version_;
        
        // Calculate totals
        stats.total_inserts = 0;
        stats.total_deletes = 0;
        
        // Fill container stats
        for (size_t i = 0; i < IdxN; ++i) {
            stats.containers[i] = container_stats_[i];
            stats.total_inserts += container_stats_[i].total_inserts;
            stats.total_deletes += container_stats_[i].total_deletes;
            stats.rotations_per_container[i] = current_versions_[i];
        }
        
        stats.op_return = op_return_stats_;
        
        return stats;
    }
    
    void print_statistics() {
        auto stats = get_statistics();
        
        log_print("\n=== UTXO Database Statistics ===\n");
        log_print("Total entries: {}\n", stats.total_entries);
        log_print("Total inserts: {}\n", stats.total_inserts);
        log_print("Total deletes: {}\n", stats.total_deletes);
        log_print("Index file rotations: {}\n", stats.index_rotations);
        
        log_print("\n--- Container Statistics ---\n");
        for (size_t i = 0; i < IdxN; ++i) {
            log_print("Container {} (size <= {} bytes):\n", i, container_sizes[i]);
            log_print("  Current entries: {}\n", stats.containers[i].current_size);
            log_print("  Total inserts: {}\n", stats.containers[i].total_inserts);
            log_print("  Total deletes: {}\n", stats.containers[i].total_deletes);
            log_print("  Failed deletes: {}\n", stats.containers[i].failed_deletes);
            log_print("  Rehash count: {}\n", stats.containers[i].rehash_count);
            log_print("  File rotations: {}\n", stats.rotations_per_container[i]);
            
            if (!stats.containers[i].value_size_distribution.empty()) {
                log_print("  Value size distribution:\n");
                for (auto const& [size, count] : stats.containers[i].value_size_distribution) {
                    log_print("    Size {}: {} entries\n", size, count);
                }
            }
        }
        
        log_print("\n--- Cache Statistics ---\n");
        log_print("Cache hit rate: {:.2f}%\n", stats.cache_hit_rate * 100);
        
        log_print("\n--- Search Performance ---\n");
        log_print("Total operations: {}\n", stats.search_summary.total_operations);
        log_print("Hit rate: {:.2f}%\n", stats.search_summary.hit_rate * 100);
        log_print("Current version hits: {:.2f}%\n", 
                 stats.search_summary.total_operations > 0 ? 
                 double(stats.search_summary.current_version_hits) / stats.search_summary.total_operations * 100 : 0);
        log_print("Average depth: {:.2f}\n", stats.search_summary.avg_depth);
        log_print("Average UTXO age: {:.2f} blocks\n", stats.search_summary.avg_utxo_age);
        log_print("Cache hit rate (when depth > 0): {:.2f}%\n", stats.search_summary.cache_hit_rate * 100);
        
        log_print("\n--- OP_RETURN Set Statistics ---\n");
        log_print("Total OP_RETURNs inserted: {}\n", stats.op_return.total_inserts);
        log_print("Total OP_RETURNs deleted: {}\n", stats.op_return.total_deletes);
        log_print("Current OP_RETURNs size: {}\n", stats.op_return.current_size);
        log_print("Failed OP_RETURN deletes: {}\n", stats.op_return.failed_deletes);

        log_print("\n================================\n");
    }
    
    void reset_all_statistics() {
        for (auto& cs : container_stats_) {
            cs = container_stats{};
        }
        op_return_stats_ = op_return_stats{};
        reset_search_stats();
    }

private:
    static constexpr std::string_view data_file_format = "{}/cont_{}_v{:05}.dat";
    static constexpr std::string_view index_file_format = "{}/index_v{:05}.dat";
    
    // Storage
    fs::path db_path_ = "utxo_interprocess";
    
    // Index storage
    std::unique_ptr<bip::managed_mapped_file> index_segment_;
    index_map_t* index_ = nullptr;
    size_t current_index_version_ = 0;
    std::vector<index_metadata> index_metadata_;
    size_t index_min_buckets_ok_ = 0;
    
    // Data storage
    std::array<std::unique_ptr<bip::managed_mapped_file>, IdxN> segments_;
    std::array<void*, IdxN> containers_{};
    std::array<size_t, IdxN> current_versions_ = {};
    std::array<size_t, IdxN> min_buckets_ok_ = {};
    
    // Metadata and caching
    std::array<std::vector<file_metadata>, IdxN> file_metadata_;
    file_cache file_cache_ = file_cache("");
    search_stats search_stats_;
    size_t entries_count_ = 0;
    
    // OP_RETURN set storage
    std::unique_ptr<bip::managed_mapped_file> op_return_segment_;
    op_return_set_t* op_return_set_ = nullptr;
    
    // Statistics
    std::array<container_stats, IdxN> container_stats_;
    op_return_stats op_return_stats_;

    // Get container
    template <size_t Index>
    utxo_map<container_sizes[Index]>& container() {
        return *static_cast<utxo_map<container_sizes[Index]>*>(containers_[Index]);
    }
    
    template <size_t Index>
    utxo_map<container_sizes[Index]> const& container() const {
        return *static_cast<utxo_map<container_sizes[Index]> const*>(containers_[Index]);
    }
    
    // Find in index (searches all index versions from newest to oldest)
    std::optional<index_pointer> find_in_index(utxo_key_t const& key, uint32_t height, char op) {
        // Try current index version first
        if (auto it = index_->find(key); it != index_->end()) {
            search_stats_.add_record(height, 0, 0, false, true, op);
            return it->second;
        }
        
        // Search previous index versions
        for (size_t v = current_index_version_; v-- > 0;) {
            try {
                auto [idx_map, cache_hit] = file_cache_.get_or_open_index_file(v);
                
                if (auto it = idx_map.find(key); it != idx_map.end()) {
                    size_t depth = current_index_version_ - v;
                    search_stats_.add_record(height, 0, depth, cache_hit, true, op);
                    return it->second;
                }
            } catch (std::exception const& e) {
                log_print("Error accessing index version {}: {}\n", v, e.what());
            }
        }
        
        search_stats_.add_record(height, 0, current_index_version_ + 1, false, false, op);
        return std::nullopt;
    }
    
    // Erase from index (only from current version)
    void erase_from_index(utxo_key_t const& key) {
        index_->erase(key);
        update_index_metadata_on_delete();
    }
    
    // Insert in container
    template <size_t Index>
    bool insert_in_container(utxo_key_t const& key, span_bytes value, uint32_t height) {
        // Check if rotation needed
        if (!can_insert_safely<Index>()) {
            log_print("Rotating container {} due to safety constraints\n", Index);
            new_version<Index>();
        }
        
        // Allocate index for this entry
        uint64_t entry_index = file_metadata_[Index][current_versions_[Index]].allocate_index();
        
        // Create index pointer
        index_pointer ptr(Index, current_versions_[Index], entry_index);
        
        // Check if index rotation needed
        if (!can_insert_index_safely()) {
            log_print("Rotating index due to safety constraints\n");
            new_index_version();
        }
        
        // Insert into index first
        auto [idx_it, idx_inserted] = index_->emplace(key, ptr);
        if (!idx_inserted) {
            log_print("Key already exists in index!\n");
            return false;
        }
        
        // Prepare value
        utxo_value<container_sizes[Index]> val;
        val.block_height = height;
        val.set_data(value);
        
        // Insert into data container
        try {
            auto& map = container<Index>();
            
            size_t bucket_count_before = map.bucket_count();
            
            auto [it, inserted] = map.emplace(entry_index, val);
            if (inserted) {
                ++entries_count_;
                ++container_stats_[Index].total_inserts;
                ++container_stats_[Index].current_size;
                ++container_stats_[Index].value_size_distribution[value.size()];
                
                if (map.bucket_count() != bucket_count_before) {
                    ++container_stats_[Index].rehash_count;
                }
                
                update_metadata_on_insert(Index, current_versions_[Index], height);
                update_index_metadata_on_insert();
                return true;
            } else {
                // Rollback index insertion
                index_->erase(key);
                log_print("Failed to insert into data container!\n");
                return false;
            }
        } catch (boost::interprocess::bad_alloc const& e) {
            // Rollback index insertion
            index_->erase(key);
            
            log_print("Error inserting into container {}: {}\n", Index, e.what());
            log_print("Rotating container and retrying...\n");
            
            new_version<Index>();
            
            // Retry with recursive call
            return insert_in_container<Index>(key, value, height);
        }
    }
    
    // Find in data file
    template <size_t Index>
    std::optional<std::vector<uint8_t>> find_in_data_file(index_pointer ptr, uint32_t height) {
        try {
            if (ptr.version() == current_versions_[Index]) {
                // Current version
                auto& map = container<Index>();
                if (auto it = map.find(ptr.index()); it != map.end()) {
                    auto data = it->second.get_data();
                    return std::vector<uint8_t>(data.begin(), data.end());
                }
            } else {
                // Previous version
                auto [map, cache_hit] = file_cache_.get_or_open_data_file<Index>(Index, ptr.version());
                if (auto it = map.find(ptr.index()); it != map.end()) {
                    auto data = it->second.get_data();
                    return std::vector<uint8_t>(data.begin(), data.end());
                }
            }
        } catch (std::exception const& e) {
            log_print("Error accessing data file ({}, v{}): {}\n", Index, ptr.version(), e.what());
        }
        
        log_print("ERROR: Index pointed to non-existent data! DB might be corrupted.\n");
        return std::nullopt;
    }
    
    // Erase from data file
    template <size_t Index>
    bool erase_from_data_file(index_pointer ptr, uint32_t height) {
        try {
            if (ptr.version() == current_versions_[Index]) {
                // Current version
                auto& map = container<Index>();
                if (map.erase(ptr.index()) > 0) {
                    --container_stats_[Index].current_size;
                    ++container_stats_[Index].total_deletes;
                    update_metadata_on_delete(Index, ptr.version());
                    return true;
                }
            } else {
                // Previous version
                auto [map, cache_hit] = file_cache_.get_or_open_data_file<Index>(Index, ptr.version());
                if (map.erase(ptr.index()) > 0) {
                    --container_stats_[Index].current_size;
                    ++container_stats_[Index].total_deletes;
                    update_metadata_on_delete(Index, ptr.version());
                    return true;
                }
            }
        } catch (std::exception const& e) {
            log_print("Error accessing data file ({}, v{}): {}\n", Index, ptr.version(), e.what());
        }
        
        return false;
    }
    
    // Index management
    void open_or_create_index(size_t version) {
        auto file_name = fmt::format(index_file_format, db_path_.string(), version);
        
        index_segment_ = std::make_unique<bip::managed_mapped_file>(
            bip::open_or_create, file_name.c_str(), index_file_size);
        
        index_ = index_segment_->find_or_construct<index_map_t>("index_map")(
            index_min_buckets_ok_,
            key_hash{},
            key_equal{},
            index_segment_->get_allocator<typename index_map_t::value_type>()
        );
        
        current_index_version_ = version;
    }
    
    void close_index() {
        if (index_segment_) {
            save_index_metadata_to_disk(current_index_version_);
            index_segment_->flush();
            index_segment_.reset();
            index_ = nullptr;
        }
    }
    
    void new_index_version() {
        close_index();
        ++current_index_version_;
        
        if (index_metadata_.size() <= current_index_version_) {
            index_metadata_.resize(current_index_version_ + 1);
        }
        index_metadata_[current_index_version_] = index_metadata{};
        
        open_or_create_index(current_index_version_);
        log_print("Index rotated to version {}\n", current_index_version_);
    }
    
    bool can_insert_index_safely() const {
        if (index_->bucket_count() > 0) {
            float next_load = float(index_->size() + 1) / float(index_->bucket_count());
            if (next_load >= index_->max_load_factor() * 0.95f) {
                return false;
            }
        }
        
        if (index_segment_) {
            try {
                size_t free_memory = index_segment_->get_free_memory();
                size_t entry_size = sizeof(typename index_map_t::value_type);
                size_t buffer_size = entry_size * 10;
                
                return free_memory > buffer_size;
            } catch (...) {
                return false;
            }
        }
        
        return true;
    }
    
    // Container management
    template <size_t Index>
    void open_or_create_container(size_t version) {
        auto file_name = fmt::format(data_file_format, db_path_.string(), Index, version);
        
        segments_[Index] = std::make_unique<bip::managed_mapped_file>(
            bip::open_or_create, file_name.c_str(), file_sizes[Index]);
        
        auto* segment = segments_[Index].get();
        containers_[Index] = segment->find_or_construct<utxo_map<container_sizes[Index]>>("db_map")(
            min_buckets_ok_[Index],
            boost::hash<data_key_t>{},
            std::equal_to<data_key_t>{},
            segment->get_allocator<typename utxo_map<container_sizes[Index]>::value_type>()
        );
        
        current_versions_[Index] = version;
    }
    
    template <size_t Index>
    void close_container() {
        if (segments_[Index]) {
            save_metadata_to_disk(Index, current_versions_[Index]);
            segments_[Index]->flush();
            segments_[Index].reset();
            containers_[Index] = nullptr;
        }
    }
    
    template <size_t Index>
    void new_version() {
        close_container<Index>();
        ++current_versions_[Index];
        
        if (file_metadata_[Index].size() <= current_versions_[Index]) {
            file_metadata_[Index].resize(current_versions_[Index] + 1);
        }
        file_metadata_[Index][current_versions_[Index]] = file_metadata{};
        file_metadata_[Index][current_versions_[Index]].container_index = Index;
        file_metadata_[Index][current_versions_[Index]].version = current_versions_[Index];
        
        open_or_create_container<Index>(current_versions_[Index]);
        log_print("Container {} rotated to version {}\n", Index, current_versions_[Index]);
    }
    
    // Metadata management
    void update_metadata_on_insert(size_t index, size_t version, uint32_t height) {
        if (file_metadata_[index].size() <= version) {
            file_metadata_[index].resize(version + 1);
        }
        file_metadata_[index][version].update_on_insert(height);
    }
    
    void update_metadata_on_delete(size_t index, size_t version) {
        if (file_metadata_[index].size() > version) {
            file_metadata_[index][version].update_on_delete();
        }
    }
    
    void update_index_metadata_on_insert() {
        if (index_metadata_.size() <= current_index_version_) {
            index_metadata_.resize(current_index_version_ + 1);
        }
        index_metadata_[current_index_version_].entry_count++;
    }
    
    void update_index_metadata_on_delete() {
        if (index_metadata_.size() > current_index_version_) {
            index_metadata_[current_index_version_].entry_count--;
        }
    }
    
    void save_metadata_to_disk(size_t index, size_t version) {
        auto metadata_file = fmt::format("{}/meta_{}_{:05}.dat", db_path_.string(), index, version);
        // TODO: Implement actual saving
    }
    
    void load_metadata_from_disk(size_t index, size_t version) {
        auto metadata_file = fmt::format("{}/meta_{}_{:05}.dat", db_path_.string(), index, version);
        // TODO: Implement actual loading
    }
    
    void save_index_metadata_to_disk(size_t version) {
        auto metadata_file = fmt::format("{}/index_meta_{:05}.dat", db_path_.string(), version);
        // TODO: Implement actual saving
    }
    
    void load_index_metadata_from_disk(size_t version) {
        auto metadata_file = fmt::format("{}/index_meta_{:05}.dat", db_path_.string(), version);
        // TODO: Implement actual loading
    }
    
    // OP_RETURN management
    void open_or_create_op_return_set() {
        auto file_path = db_path_ / op_return_file_name;
        bool new_file = !fs::exists(file_path);

        try {
            size_t op_return_file_size = 400_mib;

            op_return_segment_ = std::make_unique<bip::managed_mapped_file>(
                bip::open_or_create, file_path.string().c_str(), op_return_file_size);

            if (new_file) {
                op_return_set_ = op_return_segment_->construct<op_return_set_t>("OPReturnSet")(
                    op_return_segment_->get_segment_manager());
                log_print("Created new OP_RETURN set file: {}\n", file_path.string());
            } else {
                op_return_set_ = op_return_segment_->find_or_construct<op_return_set_t>("OPReturnSet")(
                    op_return_segment_->get_segment_manager());
                log_print("Opened existing OP_RETURN set file: {}\n", file_path.string());
            }
            op_return_stats_.current_size = op_return_set_ ? op_return_set_->size() : 0;
        } catch (bip::interprocess_exception const& e) {
            log_print("ERROR: Failed to open or create OP_RETURN set file {}: {}\n", 
                     file_path.string(), e.what());
            op_return_segment_.reset();
            op_return_set_ = nullptr;
            throw;
        }
    }

    void close_op_return_set() {
        if (op_return_segment_) {
            op_return_segment_.reset();
            op_return_set_ = nullptr;
            log_print("Closed OP_RETURN set file.\n");
        }
    }
    
    // Utilities
    size_t get_index_from_size(size_t size) const {
        for (size_t i = 0; i < IdxN; ++i) {
            if (size <= container_sizes[i]) return i;
        }
        return IdxN;
    }
    
    size_t find_latest_version_from_files(size_t index) {
        size_t version = 0;
        while (fs::exists(fmt::format(data_file_format, db_path_.string(), index, version))) {
            ++version;
        }
        return version > 0 ? version - 1 : 0;
    }
    
    size_t find_latest_index_version_from_files() {
        size_t version = 0;
        while (fs::exists(fmt::format(index_file_format, db_path_.string(), version))) {
            ++version;
        }
        return version > 0 ? version - 1 : 0;
    }
    
    template <size_t Index>
    bool can_insert_safely() const {
        auto const& map = container<Index>();
        
        if (map.bucket_count() > 0) {
            float next_load = float(map.size() + 1) / float(map.bucket_count());
            if (next_load >= map.max_load_factor() * 0.95f) {
                return false;
            }
        }
        
        if (segments_[Index]) {
            try {
                size_t free_memory = segments_[Index]->get_free_memory();
                size_t entry_size = sizeof(typename utxo_map<container_sizes[Index]>::value_type);
                size_t buffer_size = entry_size * 10;
                
                return free_memory > buffer_size;
            } catch (...) {
                return false;
            }
        }
        
        return true;
    }
    
    template <size_t Index>
    size_t find_optimal_buckets(std::string const& file_path, size_t file_size, size_t initial_buckets) {
        log_print("Finding optimal number of buckets for file: {} (size: {})...\n", file_path, file_size);

        size_t left = 1;
        size_t right = initial_buckets;
        size_t best_buckets = left;
        
        while (left <= right) {
            size_t mid = left + (right - left) / 2;
            log_print("Trying with {} buckets...\n", mid);

            std::string temp_file = fmt::format("{}/temp_{}_{}.dat", file_path, file_size, mid);
            try {
                bip::managed_mapped_file segment(bip::open_or_create, temp_file.c_str(), file_size);

                using temp_map_t = utxo_map<container_sizes[Index]>;
                auto* map = segment.find_or_construct<temp_map_t>("temp_map")(
                    mid,
                    boost::hash<data_key_t>{},
                    std::equal_to<data_key_t>{},
                    segment.get_allocator<std::pair<data_key_t const, utxo_value<container_sizes[Index]>>>()
                );

                best_buckets = mid;
                left = mid + 1;
                log_print("Buckets {} successful. Increasing range...\n", mid);
            } catch (boost::interprocess::bad_alloc const& e) {
                log_print("Failed with {} buckets: {}\n", mid, e.what());
                right = mid - 1;
            }

            std::filesystem::remove(temp_file);
        }

        log_print("Optimal number of buckets: {}\n", best_buckets);
        return best_buckets;
    }
    
    size_t find_optimal_index_buckets(std::string const& file_path, size_t file_size, size_t initial_buckets) {
        log_print("Finding optimal number of buckets for index file: {} (size: {})...\n", file_path, file_size);

        size_t left = 1;
        size_t right = initial_buckets;
        size_t best_buckets = left;
        
        while (left <= right) {
            size_t mid = left + (right - left) / 2;
            log_print("Trying with {} buckets...\n", mid);

            std::string temp_file = fmt::format("{}/temp_index_{}_{}.dat", file_path, file_size, mid);
            try {
                bip::managed_mapped_file segment(bip::open_or_create, temp_file.c_str(), file_size);

                auto* map = segment.find_or_construct<index_map_t>("temp_index")(
                    mid,
                    key_hash{},
                    key_equal{},
                    segment.get_allocator<std::pair<utxo_key_t const, index_pointer>>()
                );

                best_buckets = mid;
                left = mid + 1;
                log_print("Buckets {} successful. Increasing range...\n", mid);
            } catch (boost::interprocess::bad_alloc const& e) {
                log_print("Failed with {} buckets: {}\n", mid, e.what());
                right = mid - 1;
            }

            std::filesystem::remove(temp_file);
        }

        log_print("Optimal number of buckets: {}\n", best_buckets);
        return best_buckets;
    }
};

} // namespace utxo