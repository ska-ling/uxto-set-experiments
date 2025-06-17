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

// Constants - KEEPING YOUR EXACT VALUES
inline constexpr size_t utxo_key_size = 36;
inline constexpr std::array<size_t, 4> container_sizes = {44, 128, 512, 10240};
inline constexpr std::array<size_t, 4> file_sizes = {
    400_mib,
    100_mib,
    100_mib,
    100_mib
};
inline constexpr size_t index_file_size = 800_mib; // Size for index files


using index_value_t = uint64_t;


// Index entry structure
// 64 bits total: 2 bits container + 16 bits version + 46 bits local index
struct index_entry {
    // uint64_t value;
    
    static constexpr uint64_t CONTAINER_BITS = 2;
    static constexpr uint64_t VERSION_BITS = 16;
    static constexpr uint64_t INDEX_BITS = 46;
    
    static constexpr uint64_t CONTAINER_MASK = (1ULL << CONTAINER_BITS) - 1;
    static constexpr uint64_t VERSION_MASK = (1ULL << VERSION_BITS) - 1;
    static constexpr uint64_t INDEX_MASK = (1ULL << INDEX_BITS) - 1;
    
    index_entry() = delete;

    static constexpr
    index_value_t to_index_value(uint8_t container, uint16_t version, uint64_t local_index) {
        uint64_t value;
        value = (uint64_t(container) & CONTAINER_MASK) << (VERSION_BITS + INDEX_BITS);
        value |= (uint64_t(version) & VERSION_MASK) << INDEX_BITS;
        value |= (local_index & INDEX_MASK);
        return value;
    }

    static constexpr
    uint8_t get_container(uint64_t value) {
        return uint8_t((value >> (VERSION_BITS + INDEX_BITS)) & CONTAINER_MASK);
    }
    
    static constexpr
    uint16_t get_version(uint64_t value) {
        return uint16_t((value >> INDEX_BITS) & VERSION_MASK);
    }
    
    static constexpr
    uint64_t get_local_index(uint64_t value) {
        return value & INDEX_MASK;
    }

    static constexpr
    std::tuple<uint8_t, uint16_t, uint64_t> get_triple(uint64_t value) {
        return std::make_tuple(
            get_container(value),
            get_version(value),
            get_local_index(value)
        );
    }



    // index_entry(uint8_t container, uint16_t version, uint64_t local_index) {
    //     value = (static_cast<uint64_t>(container) & CONTAINER_MASK) << (VERSION_BITS + INDEX_BITS);
    //     value |= (static_cast<uint64_t>(version) & VERSION_MASK) << INDEX_BITS;
    //     value |= (local_index & INDEX_MASK);
    // }



    // index_entry() : value(0) {}
    // index_entry(uint8_t container, uint16_t version, uint64_t local_index) {
    //     value = (static_cast<uint64_t>(container) & CONTAINER_MASK) << (VERSION_BITS + INDEX_BITS);
    //     value |= (static_cast<uint64_t>(version) & VERSION_MASK) << INDEX_BITS;
    //     value |= (local_index & INDEX_MASK);
    // }
    
    // uint8_t get_container() const {
    //     return static_cast<uint8_t>((value >> (VERSION_BITS + INDEX_BITS)) & CONTAINER_MASK);
    // }
    
    // uint16_t get_version() const {
    //     return static_cast<uint16_t>((value >> INDEX_BITS) & VERSION_MASK);
    // }
    
    // uint64_t get_local_index() const {
    //     return value & INDEX_MASK;
    // }
};




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

// Types
using utxo_key_t = std::array<uint8_t, utxo_key_size>;
using data_key_t = uint64_t; // Key in data files is now the local index

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
    index_value_t,
    key_hash,
    key_equal,
    bip::allocator<std::pair<utxo_key_t const, index_value_t>, segment_manager_t>
>;

// Select appropriate uint type for size
template <size_t Size>
using size_type = std::conditional_t<Size <= 255, uint8_t, uint16_t>;

// Simplified value structure (without key now, since we use local index)
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

// Data map type - now uses local index as key
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

// File metadata - keeping your exact structure
struct file_metadata {
    uint32_t min_block_height = UINT32_MAX;
    uint32_t max_block_height = 0;
    utxo_key_t min_key;
    utxo_key_t max_key;
    size_t entry_count = 0;
    size_t container_index = 0;
    size_t version = 0;
    
    file_metadata() {
        min_key.fill(0xFF);
        max_key.fill(0x00);
    }
    
    bool key_in_range(utxo_key_t const& key) const {
        return entry_count > 0 && 
            key >= min_key && key <= max_key;
    }
    
    bool block_in_range(uint32_t height) const {
        return entry_count > 0 && 
            height >= min_block_height && height <= max_block_height;
    }
    
    void update_on_insert(utxo_key_t const& key, uint32_t height) {
        if (entry_count == 0) {
            min_key = max_key = key;
            min_block_height = max_block_height = height;
        } else {
            if (key < min_key) min_key = key;
            if (key > max_key) max_key = key;
            min_block_height = std::min(min_block_height, height);
            max_block_height = std::max(max_block_height, height);
        }
        entry_count++;
    }
    
    void update_on_delete() {
        if (entry_count > 0) --entry_count;
    }
};

// Simplified search record
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

// Search statistics with cleaner interface
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

// Deferred deletion entry
struct deferred_deletion_entry {
    utxo_key_t key;
    uint32_t height;
    
    deferred_deletion_entry(utxo_key_t const& k, uint32_t h) 
        : key(k), height(h) {}
    
    // Equality and hash for use in unordered_flat_set
    bool operator==(deferred_deletion_entry const& other) const {
        return key == other.key;
    }
    
    friend std::size_t hash_value(deferred_deletion_entry const& entry) {
        return boost::hash<utxo_key_t>{}(entry.key);
    }
};

// File cache for both index and data files
struct file_cache {
    enum class file_type { INDEX, DATA };
    
    struct cache_key {
        file_type type;
        size_t container_index; // For data files, ignored for index files
        size_t version;
        
        bool operator==(cache_key const& other) const {
            return type == other.type && 
                   container_index == other.container_index && 
                   version == other.version;
        }
    };
    
    struct cache_key_hash {
        std::size_t operator()(cache_key const& k) const {
            std::size_t seed = 0;
            boost::hash_combine(seed, static_cast<int>(k.type));
            boost::hash_combine(seed, k.container_index);
            boost::hash_combine(seed, k.version);
            return seed;
        }
    };

    explicit 
    file_cache(std::string const& path, size_t max_size = 5) 
        : base_path_(path), 
          max_cached_files_(max_size)
    {}
    
    index_map_t& get_or_open_index_file(size_t version) {
        cache_key key{file_type::INDEX, 0, version};
        
        ++gets_;
        auto const now = std::chrono::steady_clock::now();
        
        // Track access
        ++access_frequency_[key];
        
        // Check cache
        if (auto it = cache_.find(key); it != cache_.end()) {
            it->second.last_used = now;
            ++it->second.access_count;
            ++hits_;
            return *static_cast<index_map_t*>(it->second.map_ptr);
        }
        
        // Evict if needed
        if (cache_.size() >= max_cached_files_) {
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
            
            cache_[key] = {
                .segment = std::move(segment),
                .map_ptr = map,
                .last_used = now,
                .access_count = 1,
                .is_pinned = false
            };
            
            return *map;
        } catch (std::exception const& e) {
            auto file_path = make_index_file_path(version);
            log_print("file_cache: ERROR opening index {}: {}\n", file_path, e.what());
            throw;
        }
    }
    
    template <size_t Index>
    utxo_map<container_sizes[Index]>& get_or_open_data_file(size_t version) {
        cache_key key{file_type::DATA, Index, version};
        
        ++gets_;
        auto const now = std::chrono::steady_clock::now();
        
        // Track access
        ++access_frequency_[key];
        
        // Check cache
        if (auto it = cache_.find(key); it != cache_.end()) {
            it->second.last_used = now;
            ++it->second.access_count;
            ++hits_;
            return *static_cast<utxo_map<container_sizes[Index]>*>(it->second.map_ptr);
        }
        
        // Evict if needed
        if (cache_.size() >= max_cached_files_) {
            evict_lru();
        }
        
        // Open file
        try {
            auto file_path = make_data_file_path(Index, version);
            auto segment = std::make_unique<bip::managed_mapped_file>(
                bip::open_only, file_path.c_str());
            
            auto* map = segment->find<utxo_map<container_sizes[Index]>>("db_map").first;
            if (!map) {
                throw std::runtime_error("Data map not found in file");
            }
            
            cache_[key] = {
                .segment = std::move(segment),
                .map_ptr = map,
                .last_used = now,
                .access_count = 1,
                .is_pinned = false
            };
            
            return *map;
        } catch (std::exception const& e) {
            auto file_path = make_data_file_path(Index, version);
            log_print("file_cache: ERROR opening data {}: {}\n", file_path, e.what());
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
    
    std::string make_index_file_path(size_t version) const {
        return fmt::format("{}/index_v{:05}.dat", base_path_, version);
    }
    
    std::string make_data_file_path(size_t container, size_t version) const {
        return fmt::format("{}/cont_{}_v{:05}.dat", base_path_, container, version);
    }
    
    void evict_lru() {
        auto lru = std::ranges::min_element(cache_,
            [](auto const& a, auto const& b) {
                if (a.second.is_pinned != b.second.is_pinned)
                    return !a.second.is_pinned;
                return a.second.last_used < b.second.last_used;
            });
        
        if (lru != cache_.end()) {
            cache_.erase(lru);
            ++evictions_;
        }
    }
    
    boost::unordered_flat_map<cache_key, cached_file, cache_key_hash> cache_;
    boost::unordered_flat_map<cache_key, size_t, cache_key_hash> access_frequency_;
    size_t max_cached_files_;
    size_t gets_ = 0;
    size_t hits_ = 0;
    size_t evictions_ = 0;
    std::string base_path_;
};

// Main database class
class utxo_db {
    using span_bytes = std::span<uint8_t const>;
    static constexpr auto IdxN = container_sizes.size();
    static constexpr std::string_view op_return_file_name = "op_return_set.dat";
    static constexpr std::string_view op_return_metadata_name = "op_return_meta.dat";

    // Statistics structures (keeping the same as before)
    struct container_stats {
        size_t total_inserts = 0;
        size_t total_deletes = 0;
        size_t current_size = 0;
        size_t failed_deletes = 0;
        size_t deferred_deletes = 0;
        size_t rehash_count = 0;
        std::map<size_t, size_t> value_size_distribution;
    };
    
    struct deferred_stats {
        size_t total_deferred = 0;
        size_t successfully_processed = 0;
        size_t failed_to_delete = 0;
        size_t max_queue_size = 0;
        size_t processing_runs = 0;
        std::chrono::milliseconds total_processing_time{0};
        std::map<size_t, size_t> deletions_by_depth;
    };
    
    struct not_found_stats {
        size_t total_not_found = 0;
        size_t total_search_depth = 0;
        size_t max_search_depth = 0;
        std::map<size_t, size_t> depth_distribution;
    };
    
    struct utxo_lifetime_stats {
        std::map<uint32_t, size_t> age_distribution;
        uint32_t max_age = 0;
        double average_age = 0.0;
        size_t total_spent = 0;
    };
    
    struct fragmentation_stats {
        std::array<double, IdxN> fill_ratios{};
        std::array<size_t, IdxN> wasted_space{};
    };

    struct op_return_stats {
        size_t total_inserts = 0;
        size_t total_deletes = 0;
        size_t current_size = 0;
        size_t failed_deletes = 0;
    };

public:
    void configure(std::string_view path, bool remove_existing = false) {
        db_path_ = path;
        
        if (remove_existing && fs::exists(path)) {
            fs::remove_all(path);
        }
        fs::create_directories(path);
        
        // Configure file cache with base path
        file_cache_ = file_cache(std::string(path));
        
        // Find optimal buckets for containers
        static_assert(IdxN == 4);
        min_buckets_ok_[0] = find_optimal_buckets<0>("./optimal", file_sizes[0], 7864304);
        log_print("Optimal number of buckets for container {} and file size {}: {}\n", 0, file_sizes[0], min_buckets_ok_[0]);
        min_buckets_ok_[1] = find_optimal_buckets<1>("./optimal", file_sizes[1], 7864304);
        log_print("Optimal number of buckets for container {} and file size {}: {}\n", 1, file_sizes[1], min_buckets_ok_[1]);
        min_buckets_ok_[2] = find_optimal_buckets<2>("./optimal", file_sizes[2], 7864304);
        log_print("Optimal number of buckets for container {} and file size {}: {}\n", 2, file_sizes[2], min_buckets_ok_[2]);
        min_buckets_ok_[3] = find_optimal_buckets<3>("./optimal", file_sizes[3], 7864304);
        log_print("Optimal number of buckets for container {} and file size {}: {}\n", 3, file_sizes[3], min_buckets_ok_[3]);

        // Initialize index
        size_t latest_index_version = find_latest_index_version();
        open_or_create_index(latest_index_version);

        // Initialize containers
        for_each_index<IdxN>([&](auto I) {
            size_t latest_version = find_latest_version_from_files(I);
            open_or_create_container<I>(latest_version);
            
            // Reset next local index for each container
            next_local_index_[I] = find_max_local_index_in_container<I>(latest_version) + 1;
            
            // Load metadata
            for (size_t v = 0; v <= latest_version; ++v) {
                load_metadata_from_disk(I, v);
            }
        });

        open_or_create_op_return_set();
    }
    
    void close() {
        // Close index
        if (index_segment_) {
            save_index_metadata();
            index_segment_->flush();
            index_segment_.reset();
            current_index_ = nullptr;
        }
        
        // Close data containers
        for_each_index<IdxN>([&](auto I) {
            close_container<I>();
        });
        
        close_op_return_set();
    }
    
    size_t size() const {
        return entries_count_;
    }

    // Clean insert interface
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

    // Insert OP_RETURN keys
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
    
    // Clean erase interface with deferred deletion
    size_t erase(utxo_key_t const& key, uint32_t height) {
        // Try current index first
        if (!current_index_) {
            log_print("ERROR: No current index available\n");
            return 0;
        }
        
        auto it = current_index_->find(key);
        if (it != current_index_->end()) {
            // Found in current index - extract location and erase from data
            auto const entry = it->second;
            current_index_->erase(it);
            
            // Erase from data file
            size_t result = erase_from_data_file(entry, height);
            if (result > 0) {
                entries_count_ -= result;
                return result;
            } else {
                log_print("ERROR: Index pointed to non-existent data entry - DB corrupted!\n");
                return 0;
            }
        }
        
        // Check OP_RETURN set
        if (op_return_set_ && op_return_set_->count(key)) {
            if (op_return_set_->erase(key)) {
                ++op_return_stats_.total_deletes;
                --op_return_stats_.current_size;
                return 1; 
            }
        }
        
        // Not found in current index - add to deferred
        ++not_found_stats_.total_not_found;
        add_to_deferred_deletions(key, height);
        return 0;
    }
    
    // Clean find interface
    std::optional<std::vector<uint8_t>> find(utxo_key_t const& key, uint32_t height) {
        // Try current index first
        if (!current_index_) {
            log_print("ERROR: No current index available\n");
            return std::nullopt;
        }
        
        auto it = current_index_->find(key);
        if (it != current_index_->end()) {
            // Found in current index
            auto const entry = it->second;
            return find_in_data_file(entry, height, 0);
        }
        
        // Not found - would need to search previous index versions
        // For now, we'll return nullopt and rely on deferred processing
        search_stats_.add_record(height, 0, 1, false, false, 'f');
        return std::nullopt;
    }
    
    // Process ALL deferred deletions
    std::pair<uint32_t, std::vector<utxo_key_t>> process_pending_deletions() {
        if (deferred_deletions_.empty()) return {};
        
        auto const start_time = std::chrono::steady_clock::now();
        ++deferred_stats_.processing_runs;
        
        size_t initial_size = deferred_deletions_.size();
        log_print("Processing ALL {} deferred deletions\n", initial_size);
        
        size_t successful_deletions = 0;
        std::vector<utxo_key_t> failed_deletions;
        
        // Process through previous index versions
        for (size_t idx_version = current_index_version_; idx_version-- > 0;) {
            if (deferred_deletions_.empty()) break;
            
            successful_deletions += process_deferred_in_index_version(idx_version);
        }
        
        // Collect remaining as failures
        failed_deletions.reserve(deferred_deletions_.size());
        for (auto const& entry : deferred_deletions_) {
            failed_deletions.push_back(entry.key);
        }
        
        deferred_deletions_.clear();
        
        log_print("Deferred deletion processing complete: {} successful, {} FAILED\n", 
                 successful_deletions, failed_deletions.size());
        
        auto const end_time = std::chrono::steady_clock::now();
        deferred_stats_.total_processing_time += 
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        deferred_stats_.successfully_processed += successful_deletions;
        deferred_stats_.failed_to_delete += failed_deletions.size();
        
        return {successful_deletions, std::move(failed_deletions)};
    }
    
    // Stats methods (keeping the same interface)
    search_stats const& get_search_stats() const { 
        return search_stats_; 
    }

    void reset_search_stats() { 
        search_stats_.reset(); 
    }
    
    size_t deferred_deletions_size() const {
        return deferred_deletions_.size();
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
        deferred_stats deferred;
        not_found_stats not_found;
        search_stats::summary search_summary;
        std::array<size_t, IdxN> rotations_per_container;
        std::array<size_t, IdxN> memory_usage_per_container;
        utxo_lifetime_stats lifetime;
        fragmentation_stats fragmentation;
        op_return_stats op_return;
        size_t index_rotations;
        size_t index_entries;
    };
    
    db_statistics get_statistics() {
        update_fragmentation_stats();
        
        db_statistics stats;
        
        stats.total_entries = entries_count_;
        stats.cache_hit_rate = get_cache_hit_rate();
        stats.search_summary = search_stats_.get_summary();
        stats.index_rotations = current_index_version_;
        stats.index_entries = current_index_ ? current_index_->size() : 0;
        
        stats.total_inserts = 0;
        stats.total_deletes = 0;
        
        for (size_t i = 0; i < IdxN; ++i) {
            stats.containers[i] = container_stats_[i];
            stats.total_inserts += container_stats_[i].total_inserts;
            stats.total_deletes += container_stats_[i].total_deletes;
            stats.rotations_per_container[i] = current_versions_[i];
            stats.memory_usage_per_container[i] = estimate_memory_usage(i);
        }
        
        stats.deferred = deferred_stats_;
        stats.not_found = not_found_stats_;
        stats.lifetime = lifetime_stats_;
        stats.fragmentation = fragmentation_stats_;
        stats.op_return = op_return_stats_;
        
        return stats;
    }
    
    void print_statistics() {
        auto stats = get_statistics();
        
        log_print("\n=== UTXO Database Statistics ===\n");
        log_print("Total entries: {}\n", stats.total_entries);
        log_print("Total inserts: {}\n", stats.total_inserts);
        log_print("Total deletes: {}\n", stats.total_deletes);
        log_print("Index entries: {}\n", stats.index_entries);
        log_print("Index rotations: {}\n", stats.index_rotations);
        
        log_print("\n--- Container Statistics ---\n");
        for (size_t i = 0; i < IdxN; ++i) {
            log_print("Container {} (size <= {} bytes):\n", i, container_sizes[i]);
            log_print("  Current entries: {}\n", stats.containers[i].current_size);
            log_print("  Total inserts: {}\n", stats.containers[i].total_inserts);
            log_print("  Total deletes: {}\n", stats.containers[i].total_deletes);
            log_print("  Failed deletes: {}\n", stats.containers[i].failed_deletes);
            log_print("  Deferred deletes pending: {}\n", stats.containers[i].deferred_deletes);
            log_print("  Rehash count: {}\n", stats.containers[i].rehash_count);
            log_print("  File rotations: {}\n", stats.rotations_per_container[i]);
            log_print("  Est. memory usage: {:.2f} MB\n", stats.memory_usage_per_container[i] / (1024.0*1024.0));
            log_print("  Fill ratio: {:.2f}%\n", stats.fragmentation.fill_ratios[i] * 100);
            log_print("  Wasted space: {:.2f} MB\n", stats.fragmentation.wasted_space[i] / (1024.0*1024.0));
        }
        
        log_print("\n--- Cache Statistics ---\n");
        log_print("Cache hit rate: {:.2f}%\n", stats.cache_hit_rate * 100);
        
        log_print("\n--- Search Performance ---\n");
        log_print("Total operations: {}\n", stats.search_summary.total_operations);
        log_print("Hit rate: {:.2f}%\n", stats.search_summary.hit_rate * 100);
        log_print("Average depth: {:.2f}\n", stats.search_summary.avg_depth);
        
        log_print("\n================================\n");
    }
    
    void reset_all_statistics() {
        for (auto& cs : container_stats_) {
            cs = container_stats{};
        }
        deferred_stats_ = deferred_stats{};
        not_found_stats_ = not_found_stats{};
        lifetime_stats_ = utxo_lifetime_stats{};
        fragmentation_stats_ = fragmentation_stats{};
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
    index_map_t* current_index_ = nullptr;
    size_t current_index_version_ = 0;
    size_t index_optimal_buckets_ = 15728640; // Adjust based on testing
    
    // Data storage
    std::array<std::unique_ptr<bip::managed_mapped_file>, IdxN> segments_;
    std::array<void*, IdxN> containers_{};
    std::array<size_t, IdxN> current_versions_ = {};
    std::array<size_t, IdxN> min_buckets_ok_ = {};
    std::array<uint64_t, IdxN> next_local_index_ = {}; // Next local index for each container
    
    size_t entries_count_ = 0;
    
    // Metadata and caching
    std::array<std::vector<file_metadata>, IdxN> file_metadata_;
    file_cache file_cache_ = file_cache("");
    search_stats search_stats_;
    boost::unordered_flat_set<deferred_deletion_entry> deferred_deletions_;
    
    // OP_RETURN set storage
    std::unique_ptr<bip::managed_mapped_file> op_return_segment_;
    op_return_set_t* op_return_set_ = nullptr;
    file_metadata op_return_metadata_;

    // Statistics
    std::array<container_stats, IdxN> container_stats_;
    deferred_stats deferred_stats_;
    not_found_stats not_found_stats_;
    utxo_lifetime_stats lifetime_stats_;
    fragmentation_stats fragmentation_stats_;
    op_return_stats op_return_stats_;

    // Helper to get current container
    template <size_t Index>
    utxo_map<container_sizes[Index]>& container() {
        return *static_cast<utxo_map<container_sizes[Index]>*>(containers_[Index]);
    }
    
    template <size_t Index>
    utxo_map<container_sizes[Index]> const& container() const {
        return *static_cast<utxo_map<container_sizes[Index]> const*>(containers_[Index]);
    }

    // Insert implementation with index
    template <size_t Index>
    bool insert_in_container(utxo_key_t const& key, span_bytes value, uint32_t height) {
        // Check if we need to rotate data file
        if (!can_insert_safely<Index>()) {
            log_print("Rotating container {} due to safety constraints\n", Index);
            new_data_version<Index>();
        }
        
        // Check if we need to rotate index file
        if (!can_insert_index_safely()) {
            log_print("Rotating index due to safety constraints\n");
            new_index_version();
        }
        
        // Get local index for this entry
        uint64_t local_idx = next_local_index_[Index]++;
        
        // Insert into index
        // index_entry idx_entry(Index, current_versions_[Index], local_idx);

        auto const idx_entry = index_entry::to_index_value(Index, current_versions_[Index], local_idx);

        auto [idx_it, idx_inserted] = current_index_->emplace(key, idx_entry);
        if (!idx_inserted) {
            --next_local_index_[Index]; // Rollback
            return false;
        }
        
        // Insert into data file
        utxo_value<container_sizes[Index]> val;
        val.block_height = height;
        val.set_data(value);
        
        try {
            auto& map = container<Index>();
            size_t bucket_count_before = map.bucket_count();
            
            auto [data_it, data_inserted] = map.emplace(local_idx, val);
            if (data_inserted) {
                ++entries_count_;
                ++container_stats_[Index].total_inserts;
                ++container_stats_[Index].current_size;
                ++container_stats_[Index].value_size_distribution[value.size()];
                
                if (map.bucket_count() != bucket_count_before) {
                    ++container_stats_[Index].rehash_count;
                }
                
                update_metadata_on_insert(Index, current_versions_[Index], key, height);
                return true;
            } else {
                // Rollback index entry
                current_index_->erase(idx_it);
                --next_local_index_[Index];
                return false;
            }
        } catch (boost::interprocess::bad_alloc const& e) {
            // Rollback index entry
            current_index_->erase(idx_it);
            --next_local_index_[Index];
            
            log_print("Error inserting into container {}: {}\n", Index, e.what());
            new_data_version<Index>();
            
            // Retry recursively
            return insert_in_container<Index>(key, value, height);
        }
    }
    
    // Find in data file using index entry
    std::optional<std::vector<uint8_t>> find_in_data_file(index_value_t const& entry, 
                                                          uint32_t height, 
                                                          size_t depth) {
        // auto container_idx = index_entry::get_container(entry);
        // auto version = entry.get_version();
        // auto local_idx = entry.get_local_index();

        auto const [container_idx, version, local_idx] = index_entry::get_triple(entry);
        
        auto find_with_type = [&]<size_t Index>(std::integral_constant<size_t, Index>) 
            -> std::optional<std::vector<uint8_t>> {
            
            try {
                if (version == current_versions_[Index]) {
                    // Current version
                    auto& map = container<Index>();
                    if (auto it = map.find(local_idx); it != map.end()) {
                        search_stats_.add_record(height, it->second.block_height, depth, false, true, 'f');
                        auto data = it->second.get_data();
                        return std::vector<uint8_t>(data.begin(), data.end());
                    }
                } else {
                    // Previous version - use cache
                    auto& map = file_cache_.get_or_open_data_file<Index>(version);
                    if (auto it = map.find(local_idx); it != map.end()) {
                        search_stats_.add_record(height, it->second.block_height, depth, true, true, 'f');
                        auto data = it->second.get_data();
                        return std::vector<uint8_t>(data.begin(), data.end());
                    }
                }
            } catch (std::exception const& e) {
                log_print("Error accessing data file ({}, v{}): {}\n", Index, version, e.what());
            }
            
            log_print("ERROR: Index pointed to non-existent data - DB corrupted!\n");
            search_stats_.add_record(height, 0, depth, false, false, 'f');
            return std::nullopt;
        };
        
        switch (container_idx) {
            case 0: return find_with_type(std::integral_constant<size_t, 0>{});
            case 1: return find_with_type(std::integral_constant<size_t, 1>{});
            case 2: return find_with_type(std::integral_constant<size_t, 2>{});
            case 3: return find_with_type(std::integral_constant<size_t, 3>{});
            default: return std::nullopt;
        }
    }
    
    // Erase from data file using index entry
    size_t erase_from_data_file(index_value_t const& entry, uint32_t height) {
        // auto container_idx = entry.get_container();
        // auto version = entry.get_version();
        // auto local_idx = entry.get_local_index();

        auto const [container_idx, version, local_idx] = index_entry::get_triple(entry);
        
        auto erase_with_type = [&]<size_t Index>(std::integral_constant<size_t, Index>) -> size_t {
            try {
                if (version == current_versions_[Index]) {
                    // Current version
                    auto& map = container<Index>();
                    if (auto it = map.find(local_idx); it != map.end()) {
                        uint32_t age = height - it->second.block_height;
                        update_lifetime_stats(age);
                        
                        search_stats_.add_record(height, it->second.block_height, 0, false, true, 'e');
                        map.erase(it);
                        
                        --container_stats_[Index].current_size;
                        ++container_stats_[Index].total_deletes;
                        
                        update_metadata_on_delete(Index, version);
                        return 1;
                    }
                } else {
                    // Previous version
                    auto& map = file_cache_.get_or_open_data_file<Index>(version);
                    if (auto it = map.find(local_idx); it != map.end()) {
                        uint32_t age = height - it->second.block_height;
                        update_lifetime_stats(age);
                        
                        search_stats_.add_record(height, it->second.block_height, 
                                               current_versions_[Index] - version, true, true, 'e');
                        map.erase(it);
                        
                        --container_stats_[Index].current_size;
                        ++container_stats_[Index].total_deletes;
                        
                        update_metadata_on_delete(Index, version);
                        return 1;
                    }
                }
            } catch (std::exception const& e) {
                log_print("Error erasing from data file ({}, v{}): {}\n", Index, version, e.what());
            }
            
            return 0;
        };
        
        switch (container_idx) {
            case 0: return erase_with_type(std::integral_constant<size_t, 0>{});
            case 1: return erase_with_type(std::integral_constant<size_t, 1>{});
            case 2: return erase_with_type(std::integral_constant<size_t, 2>{});
            case 3: return erase_with_type(std::integral_constant<size_t, 3>{});
            default: return 0;
        }
    }
    
    // Process deferred deletions in specific index version
    size_t process_deferred_in_index_version(size_t idx_version) {
        if (deferred_deletions_.empty()) return 0;
        
        try {
            auto& idx_map = (idx_version == current_index_version_) ? 
                           *current_index_ : 
                           file_cache_.get_or_open_index_file(idx_version);
            
            size_t successful = 0;
            auto it = deferred_deletions_.begin();
            
            while (it != deferred_deletions_.end()) {
                auto idx_it = idx_map.find(it->key);
                if (idx_it != idx_map.end()) {
                    auto const entry = idx_it->second;
                    idx_map.erase(idx_it);
                    
                    if (erase_from_data_file(entry, it->height) > 0) {
                        it = deferred_deletions_.erase(it);
                        ++successful;
                        --entries_count_;
                        
                        size_t depth = current_index_version_ - idx_version;
                        ++deferred_stats_.deletions_by_depth[depth];
                    } else {
                        log_print("ERROR: Failed to erase from data file despite index entry\n");
                        ++it;
                    }
                } else {
                    ++it;
                }
            }
            
            if (successful > 0) {
                log_print("Processed {} deletions from index version {} - {} remaining\n", 
                         successful, idx_version, deferred_deletions_.size());
            }
            
            return successful;
            
        } catch (std::exception const& e) {
            log_print("Error processing index version {}: {}\n", idx_version, e.what());
            return 0;
        }
    }
    
    // Helper functions
    void add_to_deferred_deletions(utxo_key_t const& key, uint32_t height) {
        auto [it, inserted] = deferred_deletions_.emplace(key, height);
        if (inserted) {
            ++deferred_stats_.total_deferred;
            deferred_stats_.max_queue_size = std::max(deferred_stats_.max_queue_size, 
                                                      deferred_deletions_.size());
            
            for (size_t i = 0; i < IdxN; ++i) {
                ++container_stats_[i].deferred_deletes;
            }
        }
    }
    
    void update_lifetime_stats(uint32_t age) {
        ++lifetime_stats_.age_distribution[age];
        lifetime_stats_.max_age = std::max(lifetime_stats_.max_age, age);
        ++lifetime_stats_.total_spent;
        lifetime_stats_.average_age = 
            (lifetime_stats_.average_age * (lifetime_stats_.total_spent - 1) + age) 
            / lifetime_stats_.total_spent;
    }
    
    // Index file management
    void open_or_create_index(size_t version) {
        auto file_name = fmt::format(index_file_format, db_path_.string(), version);
        
        index_segment_ = std::make_unique<bip::managed_mapped_file>(
            bip::open_or_create, file_name.c_str(), index_file_size);
        
        current_index_ = index_segment_->find_or_construct<index_map_t>("index_map")(
            index_optimal_buckets_,
            key_hash{},
            key_equal{},
            index_segment_->get_allocator<typename index_map_t::value_type>()
        );
        
        current_index_version_ = version;
    }
    
    void new_index_version() {
        if (index_segment_) {
            save_index_metadata();
            index_segment_->flush();
            index_segment_.reset();
            current_index_ = nullptr;
        }
        
        ++current_index_version_;
        open_or_create_index(current_index_version_);
        log_print("Index rotated to version {}\n", current_index_version_);
    }
    
    bool can_insert_index_safely() const {
        if (!current_index_) return false;
        
        if (current_index_->bucket_count() > 0) {
            float next_load = float(current_index_->size() + 1) / float(current_index_->bucket_count());
            if (next_load >= current_index_->max_load_factor() * 0.95f) {
                return false;
            }
        }
        
        if (index_segment_) {
            try {
                size_t free_memory = index_segment_->get_free_memory();
                size_t entry_size = sizeof(typename index_map_t::value_type);
                size_t buffer_size = entry_size * 100;
                
                return free_memory > buffer_size;
            } catch (...) {
                return false;
            }
        }
        
        return true;
    }
    
    // Data file management
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
    void new_data_version() {
        close_container<Index>();
        ++current_versions_[Index];
        next_local_index_[Index] = 0; // Reset local index for new file
        
        if (file_metadata_[Index].size() <= current_versions_[Index]) {
            file_metadata_[Index].resize(current_versions_[Index] + 1);
        }
        file_metadata_[Index][current_versions_[Index]] = file_metadata{};
        
        open_or_create_container<Index>(current_versions_[Index]);
        log_print("Container {} rotated to version {}\n", Index, current_versions_[Index]);
    }
    
    // Find max local index in container (for recovery after restart)
    template <size_t Index>
    uint64_t find_max_local_index_in_container(size_t version) {
        if (version != current_versions_[Index]) {
            // Would need to open old file - for now return 0
            return 0;
        }
        
        auto& map = container<Index>();
        uint64_t max_idx = 0;
        for (auto const& [idx, val] : map) {
            max_idx = std::max(max_idx, idx);
        }
        return max_idx;
    }
    
    // Utility functions
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
    
    size_t find_latest_index_version() {
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
    
    // Metadata management
    void update_metadata_on_insert(size_t index, size_t version, utxo_key_t const& key, uint32_t height) {
        if (file_metadata_[index].size() <= version) {
            file_metadata_[index].resize(version + 1);
        }
        file_metadata_[index][version].update_on_insert(key, height);
    }
    
    void update_metadata_on_delete(size_t index, size_t version) {
        if (file_metadata_[index].size() > version) {
            file_metadata_[index][version].update_on_delete();
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
    
    void save_index_metadata() {
        // TODO: Implement index metadata saving
    }
    
    // OP_RETURN management (keeping the same)
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
                op_return_metadata_ = file_metadata{};
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
    
    // Find optimal buckets
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
    
    void update_fragmentation_stats() {
        for_each_index<IdxN>([&](auto I) {
            auto& map = container<I>();
            if (segments_[I]) {
                try {
                    size_t total_size = file_sizes[I];
                    size_t free_memory = segments_[I]->get_free_memory();
                    size_t used_memory = total_size - free_memory;
                    
                    fragmentation_stats_.fill_ratios[I] = 
                        double(used_memory) / double(total_size);
                    
                    // Estimate wasted space (rough calculation)
                    // Usar el tipo del mapa sin decltype
                    using map_type = utxo_map<container_sizes[I]>;
                    size_t ideal_size = map.size() * sizeof(typename map_type::value_type);
                    fragmentation_stats_.wasted_space[I] = 
                        used_memory > ideal_size ? used_memory - ideal_size : 0;
                        
                } catch (...) {
                    fragmentation_stats_.fill_ratios[I] = 0.0;
                    fragmentation_stats_.wasted_space[I] = 0;
                }
            }
        });
    }
    
    size_t estimate_memory_usage(size_t index) const {
        size_t total = 0;
        
        // Current version
        if (segments_[index]) {
            total += file_sizes[index];
        }
        
        // Previous versions (estimate)
        for (size_t v = 0; v < current_versions_[index]; ++v) {
            auto const file_name = fmt::format(data_file_format, db_path_.string(), index, v);
            if (fs::exists(file_name)) {
                total += fs::file_size(file_name);
            }
        }
        
        return total;
    }
    
    size_t get_container_size(size_t index) const {
        size_t total = 0;
        
        // Dispatch to correct type
        switch (index) {
            case 0: total = container<0>().size(); break;
            case 1: total = container<1>().size(); break;
            case 2: total = container<2>().size(); break;
            case 3: total = container<3>().size(); break;
        }
        
        // Add sizes from previous versions
        if (file_metadata_[index].size() > 0) {
            for (size_t v = 0; v < current_versions_[index]; ++v) {
                if (v < file_metadata_[index].size()) {
                    total += file_metadata_[index][v].entry_count;
                }
            }
        }
        
        return total;
    }
};

} // namespace utxo