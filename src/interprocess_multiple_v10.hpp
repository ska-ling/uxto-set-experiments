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
    800_mib,
    200_mib,
    50_mib,
    50_mib
};

inline constexpr std::string_view data_file_format = "{}/cont_{}_v{:05}.dat";
inline constexpr std::string_view op_return_file_format = "{}/op_return_v{:05}.dat";


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

// void print_hex(uint8_t const* data, size_t size) {
//     // print data in hex format
//     for (size_t i = 0; i < size; ++i) {
//         fmt::print("{:02x}", data[i]);
//     }
//     fmt::print("\n");
// }

// void print_hash(kth::hash_digest hash) {
//     std::reverse(hash.begin(), hash.end()); // reverse the hash to match the expected format
//     print_hex(hash.data(), hash.size());
// }

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

// Select appropriate uint type for size
template <size_t Size>
using size_type = std::conditional_t<Size <= 255, uint8_t, uint16_t>;

// Simplified value structure (instead of tuple)
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

// Map type
template <size_t Size>
using utxo_map = boost::unordered_flat_map<
    utxo_key_t,
    utxo_value<Size>,
    key_hash,
    key_equal,
    bip::allocator<std::pair<utxo_key_t const, utxo_value<Size>>, segment_manager_t>
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

// File cache using integer pairs as keys - more efficient
struct file_cache {
    template <size_t Index>
    using ret_t = std::pair<utxo_map<container_sizes[Index]>&, bool>;
    
    using file_key_t = std::pair<size_t, size_t>; // (index, version)

    explicit 
    file_cache(std::string const& path, size_t max_size = 1) 
        : base_path_(path), 
          max_cached_files_(max_size)
    {}
    
    template <size_t Index>
    ret_t<Index> get_or_open_file(size_t index, size_t version) {
        file_key_t file_key{index, version};
        
        ++gets_;
        auto const now = std::chrono::steady_clock::now();
        
        // Track access
        ++access_frequency_[file_key];
        
        // Check cache
        if (auto it = cache_.find(file_key); it != cache_.end()) {
            it->second.last_used = now;
            ++it->second.access_count;
            ++hits_;
            return {*static_cast<utxo_map<container_sizes[Index]>*>(it->second.map_ptr), true};
        }
        
        // Evict if needed
        if (cache_.size() >= max_cached_files_) {
            evict_lru();
        }
        
        // Open file
        try {
            auto file_path = make_file_path(index, version);
            auto segment = std::make_unique<bip::managed_mapped_file>(
                bip::open_only, file_path.c_str());
            
            auto* map = segment->find<utxo_map<container_sizes[Index]>>("db_map").first;
            if (!map) {
                throw std::runtime_error("Map not found in file");
            }
            
            cache_[file_key] = {
                .segment = std::move(segment),
                .map_ptr = map,
                .last_used = now,
                .access_count = 1,
                .is_pinned = false
            };
            
            return {*map, false};
        } catch (std::exception const& e) {
            auto file_path = make_file_path(index, version);
            log_print("file_cache: ERROR opening {}: {}\n", file_path, e.what());
            throw;
        }
    }
    
    float get_hit_rate() const {
        return gets_ > 0 ? float(hits_) / float(gets_) : 0.0f;
    }
    
    void set_cache_size(size_t new_size) { 
        max_cached_files_ = new_size; 
    }
    
    // Get list of currently cached files as (index, version) pairs
    std::vector<std::pair<size_t, size_t>> get_cached_files() const {
        std::vector<std::pair<size_t, size_t>> files;
        files.reserve(cache_.size());
        for (auto const& [file_key, cached_file] : cache_) {
            files.push_back(file_key);
        }
        return files;
    }
    
    // Check if a specific file is cached by (index, version)
    bool is_cached(size_t index, size_t version) const {
        return cache_.contains(file_key_t{index, version});
    }
    
    // Get the most recently used cached file as (index, version) pair
    std::optional<std::pair<size_t, size_t>> get_most_recent_cached_file() const {
        if (cache_.empty()) return std::nullopt;
        
        auto most_recent = std::ranges::max_element(cache_,
            [](auto const& a, auto const& b) {
                return a.second.last_used < b.second.last_used;
            });
        
        return most_recent->first;
    }
        
private:
    struct cached_file {
        std::unique_ptr<bip::managed_mapped_file> segment;
        void* map_ptr;
        std::chrono::steady_clock::time_point last_used;
        size_t access_count = 0;
        bool is_pinned = false;
    };
    
    // Helper function for file path generation
    std::string make_file_path(size_t index, size_t version) const {
        return fmt::format(data_file_format, base_path_, index, version);
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
    
    boost::unordered_flat_map<file_key_t, cached_file> cache_;
    boost::unordered_flat_map<file_key_t, size_t> access_frequency_;
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
    static constexpr size_t op_return_file_size = 10_mib;

    template <size_t Index>
        requires (Index < IdxN)
    utxo_map<container_sizes[Index]>& container() {
        return *static_cast<utxo_map<container_sizes[Index]>*>(containers_[Index]);
    }

    template <size_t Index>
        requires (Index < container_sizes.size())
    utxo_map<container_sizes[Index]> const& container() const{
        return *static_cast<utxo_map<container_sizes[Index]> const*>(containers_[Index]);
    }    
      

    // Estadísticas por container
    struct container_stats {
        size_t total_inserts = 0;
        size_t total_deletes = 0;
        size_t current_size = 0;
        size_t failed_deletes = 0;
        size_t deferred_deletes = 0;
        size_t rehash_count = 0;
        std::map<size_t, size_t> value_size_distribution; // size -> count
    };
    
    // Estadísticas de deferred deletions
    struct deferred_stats {
        size_t total_deferred = 0;
        size_t successfully_processed = 0;
        size_t failed_to_delete = 0;
        size_t max_queue_size = 0;
        size_t processing_runs = 0;
        std::chrono::milliseconds total_processing_time{0};
        std::map<size_t, size_t> deletions_by_depth; // depth -> count
    };
    
    // Estadísticas de elementos no encontrados
    struct not_found_stats {
        size_t total_not_found = 0;
        size_t total_search_depth = 0;
        size_t max_search_depth = 0;
        std::map<size_t, size_t> depth_distribution; // depth -> count
    };
    
    // Estadísticas de tiempo de vida de UTXOs
    struct utxo_lifetime_stats {
        std::map<uint32_t, size_t> age_distribution; // age_in_blocks -> count
        uint32_t max_age = 0;
        double average_age = 0.0;
        size_t total_spent = 0;
    };
    
    // Estadísticas de fragmentación
    struct fragmentation_stats {
        std::array<double, IdxN> fill_ratios{};
        std::array<size_t, IdxN> wasted_space{};
    };

    // Statistics for OP_RETURN set
    struct op_return_stats {
        size_t total_inserts = 0;
        size_t total_deletes = 0;
        size_t current_size = 0;
        size_t failed_deletes = 0; // Should ideally be 0 if logic is correct
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
        
        // Initialize OP_RETURN set
        open_or_create_op_return_set();

        static_assert(IdxN == 4); // if not, we have to change the following code ...
        min_buckets_ok_[0] = find_optimal_buckets<0>("./optimal", file_sizes[0], 7864304);
        log_print("Optimal number of buckets for container {} and file size {}: {}\n", 0, file_sizes[0], min_buckets_ok_[0]);
        min_buckets_ok_[1] = find_optimal_buckets<1>("./optimal", file_sizes[1], 7864304);
        log_print("Optimal number of buckets for container {} and file size {}: {}\n", 1, file_sizes[1], min_buckets_ok_[1]);
        min_buckets_ok_[2] = find_optimal_buckets<2>("./optimal", file_sizes[2], 7864304);
        log_print("Optimal number of buckets for container {} and file size {}: {}\n", 2, file_sizes[2], min_buckets_ok_[2]);
        min_buckets_ok_[3] = find_optimal_buckets<3>("./optimal", file_sizes[3], 7864304);
        log_print("Optimal number of buckets for container {} and file size {}: {}\n", 3, file_sizes[3], min_buckets_ok_[3]);


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
        close_op_return_set();
    }
    
    size_t size() const {
        return entries_count_;
    }

    // Clean insert interface
    bool insert(utxo_key_t const& key, span_bytes value, uint32_t height) {
        size_t const index = get_index_from_size(value.size());
        if (index >= IdxN) {
            log_print("insert: Invalid index {} for value size {}. Height: {}\n", index, value.size(), height);
            print_key(key);
            log_print("Bytes: \n");
            print_hex(value.data(), value.size());
            throw std::out_of_range("Value size too large");
        }
        
        return std::visit([&](auto ic) {
            return insert_in_index<ic>(key, value, height);
        }, make_index_variant(index));
    }

    // Insert OP_RETURN keys
    void insert_op_returns(boost::unordered_flat_set<utxo_key_t> const& op_return_keys, uint32_t height) {
        if ( ! op_return_segment_ || ! op_return_set_) {
            log_print("ERROR: OP_RETURN set not initialized before insert_op_returns.\n");
            return;
        }

        for (auto const& key : op_return_keys) {
            // Check if rotation is needed before insertion
            if (op_return_needs_rotation()) {
                log_print("OP_RETURN set needs rotation due to capacity constraints\n");
                rotate_op_return_set();
            }
            
            size_t max_retries = 3;
            bool inserted = false;
            
            while (max_retries > 0 && !inserted) {
                try {
                    auto [it, success] = op_return_set_->insert(key);
                    if (success) {
                        ++op_return_stats_.total_inserts;
                        ++op_return_stats_.current_size;
                        update_op_return_metadata_on_insert(op_return_current_version_, key);
                        inserted = true;
                        
                        // Log insertion: block height and txid (first 32 bytes of key)
                        // log_print("OP_RETURN Inserted: Height: {}, Key: ", height);
                        // print_key(key); // Assuming print_key logs the key appropriately
                    } else {
                        // Key already exists, consider it as successful insertion
                        inserted = true;
                    }
                } catch (boost::interprocess::bad_alloc const& e) {
                    log_print("Error inserting OP_RETURN key: {}\n", e.what());
                    log_print("OP_RETURN set size: {}\n", op_return_set_->size());
                    
                    // Try to get memory info
                    if (op_return_segment_) {
                        try {
                            size_t free_memory = op_return_segment_->get_free_memory();
                            log_print("Free memory in OP_RETURN segment: {}\n", free_memory);
                        } catch (...) {
                            log_print("Cannot determine free memory in OP_RETURN segment\n");
                        }
                    }
                    
                    // Rotate and retry
                    rotate_op_return_set();
                    --max_retries;
                }
            }
            
            if (!inserted) {
                log_print("ERROR: Failed to insert OP_RETURN key after 3 retries\n");
                throw boost::interprocess::bad_alloc();
            }
        }
    }
    
    // Clean erase interface with deferred deletion
    size_t erase(utxo_key_t const& key, uint32_t height) {
        size_t search_depth = 0;
        
        // Try current version first
        if (auto res = erase_in_latest_version(key, height); res > 0) {
            entries_count_ -= res;
            return res;
        }
        ++search_depth;
        
        // Try cached files only
        if (auto res = erase_from_cached_files_only(key, height, search_depth); res > 0) {
            entries_count_ -= res;
            return res;
        }

        // Check OP_RETURN set (only current version) if not found in regular UTXO stores
        if (op_return_set_ && op_return_set_->count(key)) {
            if (op_return_set_->erase(key)) {
                ++op_return_stats_.total_deletes;
                --op_return_stats_.current_size;
                update_op_return_metadata_on_delete(op_return_current_version_);
                log_print("OP_RETURN Erased from current version: Height: {}, Key: ", height);
                print_key(key);
                return 1; 
            }
        }
        
        // Add to deferred deletions to be processed later
        
        // Track not found
        ++not_found_stats_.total_not_found;
        not_found_stats_.total_search_depth += search_depth;
        not_found_stats_.max_search_depth = std::max(not_found_stats_.max_search_depth, search_depth);
        ++not_found_stats_.depth_distribution[search_depth];
        
        // Defer deletion
        add_to_deferred_deletions(key, height);
        return 0;
    }
    
    // Clean find interface
    std::optional<std::vector<uint8_t>> find(utxo_key_t const& key, uint32_t height) {
        // Try current version first
        if (auto res = find_in_latest_version(key, height); res) {
            return res;
        }
        
        // Search previous versions
        return find_in_previous_versions(key, height);
    }
    
    // Stats
    search_stats const& get_search_stats() const { 
        return search_stats_; 
    }

    void reset_search_stats() { 
        search_stats_.reset(); 
    }
    
    size_t deferred_deletions_size() const {
        return deferred_deletions_.size();
    }

    // Get cache statistics
    float get_cache_hit_rate() const {
        return file_cache_.get_hit_rate();
    }
    
    // Get information about what files are currently cached
    std::vector<std::pair<size_t, size_t>> get_cached_file_info() const {
        return file_cache_.get_cached_files();
    }
    
    // Process ALL deferred deletions - must complete processing entire queue
    std::pair<uint32_t, std::vector<utxo_key_t>> process_pending_deletions() {
        if (deferred_deletions_.empty()) return {};
        
        auto const start_time = std::chrono::steady_clock::now();
        ++deferred_stats_.processing_runs;
        
        size_t initial_size = deferred_deletions_.size();
        log_print("Processing ALL {} deferred deletions - MUST complete entirely\n", initial_size);
        
        size_t successful_deletions = 0;
        
        // Phase 1: Process ALL cached files first to maximize cache efficiency
        auto cached_files = file_cache_.get_cached_files();
        if ( ! cached_files.empty()) {
            log_print("Phase 1: Processing {} cached files for {} deferred deletions...\n", 
                     cached_files.size(), deferred_deletions_.size());
            
            // Sort cached files by container index and version (most recent first within each container)
            std::ranges::sort(cached_files, [](auto const& a, auto const& b) {
                if (a.first != b.first) return a.first < b.first;
                return a.second > b.second; // Most recent version first
            });
            
            // Process each cached file
            for (auto const& [container_index, version] : cached_files) {
                if (deferred_deletions_.empty()) break;
                
                successful_deletions += process_deferred_deletions_in_file(container_index, version, true);
            }
        }
        
        // Phase 2: Process ALL remaining files systematically
        if (!deferred_deletions_.empty()) {
            log_print("Phase 2: Opening new files for {} remaining deferred deletions...\n", 
                     deferred_deletions_.size());
            
            // Get already processed versions to avoid redundant work
            std::array<std::set<size_t>, IdxN> processed_versions;
            for (auto const& [container_index, version] : cached_files) {
                processed_versions[container_index].insert(version);
            }
            
            // Process each container systematically
            for_each_index<IdxN>([&](auto I) {
                if (deferred_deletions_.empty()) return;
                if (current_versions_[I.value] == 0) return; // No previous versions
                
                // Process versions from latest-1 down to 0, skipping already processed ones
                for (size_t v = current_versions_[I.value] - 1; v != SIZE_MAX; --v) {
                    if (deferred_deletions_.empty()) break;
                    
                    // Skip if already processed in Phase 1
                    if (processed_versions[I.value].contains(v)) {
                        continue;
                    }
                    
                    auto file_name = fmt::format(data_file_format, db_path_.string(), I.value, v);
                    if (!fs::exists(file_name)) {
                        continue;
                    }
                    
                    successful_deletions += process_deferred_deletions_in_file(I.value, v, false);
                }
            });
        }
        
        // Phase 3: Process OLD OP_RETURN versions for remaining deferred deletions
        if (!deferred_deletions_.empty() && op_return_current_version_ > 0) {
            log_print("Phase 3: Processing old OP_RETURN versions for {} remaining deferred deletions...\n", 
                     deferred_deletions_.size());
            
            // Process OP_RETURN versions from latest-1 down to 0
            for (size_t v = op_return_current_version_ - 1; v != SIZE_MAX; --v) {
                if (deferred_deletions_.empty()) break;
                
                auto op_return_file = fmt::format(op_return_file_format, db_path_.string(), v);
                if (!fs::exists(op_return_file)) {
                    continue;
                }
                
                successful_deletions += process_deferred_deletions_in_op_return_file(v);
            }
        }
        
        // Collect any remaining UTXOs that couldn't be deleted (these are ERRORS)
        std::vector<utxo_key_t> failed_deletions;
        failed_deletions.reserve(deferred_deletions_.size());
        
        for (auto const& entry : deferred_deletions_) {
            failed_deletions.push_back(entry.key);
        }
        
        // Clear the deferred deletions since we've processed everything we can
        deferred_deletions_.clear();
        
        log_print("Deferred deletion processing complete: {} successful, {} FAILED (errors)\n", 
                 successful_deletions, failed_deletions.size());
        
        if ( ! failed_deletions.empty()) {
            log_print("ERROR: {} UTXOs could not be deleted - these are processing errors!\n", 
                     failed_deletions.size());
        }        
        
        // After processing
        auto const end_time = std::chrono::steady_clock::now();
        deferred_stats_.total_processing_time += 
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        deferred_stats_.successfully_processed += successful_deletions;
        deferred_stats_.failed_to_delete += failed_deletions.size();
        
        // Update failed deletes per container
        for (auto const& key : failed_deletions) {
            // We don't know the exact container, so increment all
            for (size_t i = 0; i < IdxN; ++i) {
                ++container_stats_[i].failed_deletes;
            }
        }
        
        return {successful_deletions, std::move(failed_deletions)};
    }
    

    struct db_statistics {
        // Global stats
        size_t total_entries;
        size_t total_inserts;
        size_t total_deletes;
        
        // Per container stats
        std::array<container_stats, IdxN> containers;
        
        // Cache stats
        float cache_hit_rate;
        size_t cached_files_count;
        std::vector<std::pair<size_t, size_t>> cached_files_info;
        
        // Deferred deletion stats
        deferred_stats deferred;
        
        // Not found stats
        not_found_stats not_found;
        
        // Search stats summary
        search_stats::summary search_summary;
        
        // File rotation stats
        std::array<size_t, IdxN> rotations_per_container;
        
        // Memory usage estimates
        std::array<size_t, IdxN> memory_usage_per_container;
        
        // UTXO lifetime stats
        utxo_lifetime_stats lifetime;
        
        // Fragmentation stats
        fragmentation_stats fragmentation;

        // OP_RETURN stats
        op_return_stats op_return; // Add this
    };
    
    db_statistics get_statistics() {
        // Update fragmentation stats before returning
        update_fragmentation_stats();
        
        db_statistics stats;
        
        stats.total_entries = entries_count_;
        stats.cache_hit_rate = get_cache_hit_rate();
        stats.cached_files_count = file_cache_.get_cached_files().size();
        stats.cached_files_info = get_cached_file_info();
        stats.search_summary = search_stats_.get_summary();
        
        // Calculate totals
        stats.total_inserts = 0;
        stats.total_deletes = 0;
        
        // Fill container stats
        for (size_t i = 0; i < IdxN; ++i) {
            stats.containers[i] = container_stats_[i];
            stats.total_inserts += container_stats_[i].total_inserts;
            stats.total_deletes += container_stats_[i].total_deletes;
            stats.rotations_per_container[i] = current_versions_[i]; // Assuming current_version is #rotations
            stats.memory_usage_per_container[i] = estimate_memory_usage(i);
        }
        
        stats.deferred = deferred_stats_;
        stats.not_found = not_found_stats_;
        stats.lifetime = lifetime_stats_;
        stats.fragmentation = fragmentation_stats_;
        stats.op_return = op_return_stats_; // Add this
        
        return stats;
    }
    
    void print_statistics() {
        auto stats = get_statistics();
        
        log_print("\n=== UTXO Database Statistics ===\n");
        log_print("Total entries: {}\n", stats.total_entries);
        log_print("Total inserts: {}\n", stats.total_inserts);
        log_print("Total deletes: {}\n", stats.total_deletes);
        
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
            
            // Value size distribution
            if (!stats.containers[i].value_size_distribution.empty()) {
                log_print("  Value size distribution:\n");
                for (auto const& [size, count] : stats.containers[i].value_size_distribution) {
                    log_print("    Size {}: {} entries\n", size, count);
                }
            }
        }
        
        log_print("\n--- Cache Statistics ---\n");
        log_print("Cache hit rate: {:.2f}%\n", stats.cache_hit_rate * 100);
        log_print("Cached files: {}\n", stats.cached_files_count);
        if (!stats.cached_files_info.empty()) {
            log_print("Cached file details:\n");
            for (auto const& [idx, ver] : stats.cached_files_info) {
                log_print("  Container {} version {}\n", idx, ver);
            }
        }
        
        log_print("\n--- Deferred Deletion Statistics ---\n");
        log_print("Total deferred: {}\n", stats.deferred.total_deferred);
        log_print("Successfully processed: {}\n", stats.deferred.successfully_processed);
        log_print("Failed to delete: {}\n", stats.deferred.failed_to_delete);
        log_print("Max queue size: {}\n", stats.deferred.max_queue_size);
        log_print("Processing runs: {}\n", stats.deferred.processing_runs);
        log_print("Total processing time: {} ms\n", stats.deferred.total_processing_time.count());
        if (stats.deferred.processing_runs > 0) {
            log_print("Avg time per run: {:.2f} ms\n", 
                     double(stats.deferred.total_processing_time.count()) / stats.deferred.processing_runs);
        }
        
        if (!stats.deferred.deletions_by_depth.empty()) {
            log_print("Deferred deletions by depth:\n");
            for (auto const& [depth, count] : stats.deferred.deletions_by_depth) {
                log_print("  Depth {}: {} deletions\n", depth, count);
            }
        }
        
        log_print("\n--- Not Found Statistics ---\n");
        log_print("Total not found: {}\n", stats.not_found.total_not_found);
        if (stats.not_found.total_not_found > 0) {
            log_print("Average search depth: {:.2f}\n", 
                     double(stats.not_found.total_search_depth) / stats.not_found.total_not_found);
            log_print("Max search depth: {}\n", stats.not_found.max_search_depth);
            log_print("Depth distribution:\n");
            for (auto const& [depth, count] : stats.not_found.depth_distribution) {
                log_print("  Depth {}: {} times\n", depth, count);
            }
        }
        
        log_print("\n--- UTXO Lifetime Statistics ---\n");
        log_print("Total UTXOs spent: {}\n", stats.lifetime.total_spent);
        if (stats.lifetime.total_spent > 0) {
            log_print("Average age: {:.2f} blocks\n", stats.lifetime.average_age);
            log_print("Max age: {} blocks\n", stats.lifetime.max_age);
            
            // Show age distribution (top 10)
            if (!stats.lifetime.age_distribution.empty()) {
                log_print("Age distribution (top 10):\n");
                std::vector<std::pair<uint32_t, size_t>> age_vec(
                    stats.lifetime.age_distribution.begin(), 
                    stats.lifetime.age_distribution.end()
                );
                std::partial_sort(age_vec.begin(), 
                                age_vec.begin() + std::min(size_t(10), age_vec.size()),
                                age_vec.end(),
                                [](auto const& a, auto const& b) { return a.second > b.second; });
                
                for (size_t i = 0; i < std::min(size_t(10), age_vec.size()); ++i) {
                    log_print("  Age {} blocks: {} UTXOs ({:.2f}%)\n", 
                             age_vec[i].first, 
                             age_vec[i].second,
                             double(age_vec[i].second) / stats.lifetime.total_spent * 100);
                }
            }
        }
        
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
    
    // Reset all statistics
    void reset_all_statistics() {
        for (auto& cs : container_stats_) {
            cs = container_stats{};
        }
        deferred_stats_ = deferred_stats{};
        not_found_stats_ = not_found_stats{};
        lifetime_stats_ = utxo_lifetime_stats{};
        fragmentation_stats_ = fragmentation_stats{};
        op_return_stats_ = op_return_stats{}; // Add this
        reset_search_stats();
    }    

private:
    
    // Storage
    fs::path db_path_ = "utxo_interprocess";
    std::array<std::unique_ptr<bip::managed_mapped_file>, IdxN> segments_;
    std::array<void*, IdxN> containers_{};
    std::array<size_t, IdxN> current_versions_ = {};

    std::array<size_t, IdxN> min_buckets_ok_ = {};
    size_t entries_count_ = 0; // Total entries across all containers
    
    // Metadata and caching
    std::array<std::vector<file_metadata>, IdxN> file_metadata_;
    file_cache file_cache_ = file_cache(""); // number of cached files. TODO: change
    search_stats search_stats_;
    boost::unordered_flat_set<deferred_deletion_entry> deferred_deletions_;
    
    // OP_RETURN set storage - now supports multiple versions
    std::unique_ptr<bip::managed_mapped_file> op_return_segment_;
    op_return_set_t* op_return_set_ = nullptr;
    std::vector<file_metadata> op_return_metadata_;
    size_t op_return_current_version_ = 0;



    // Agregar estos miembros a la clase
    std::array<container_stats, IdxN> container_stats_;
    deferred_stats deferred_stats_;
    not_found_stats not_found_stats_;
    utxo_lifetime_stats lifetime_stats_;
    fragmentation_stats fragmentation_stats_;

    // OP_RETURN statistics member
    op_return_stats op_return_stats_;


    // Get container
    template <size_t Index>
    utxo_map<container_sizes[Index]>& container() {
        return *static_cast<utxo_map<container_sizes[Index]>*>(containers_[Index]);
    }
    
    // Insert implementation
    template <size_t Index>
    bool insert_in_index(utxo_key_t const& key, span_bytes value, uint32_t height) {
        // Check if rotation needed
        if (!can_insert_safely<Index>()) {
            log_print("Rotating container {} due to safety constraints\n", Index);
            new_version<Index>();
        }
        
        // Insert
        utxo_value<container_sizes[Index]> val;
        val.block_height = height;
        val.set_data(value);
        
        size_t max_retries = 3;
        while (max_retries > 0) {
            try {
                auto& map = container<Index>();
                
                // Track rehashing
                size_t bucket_count_before = map.bucket_count();
                
                auto [it, inserted] = map.emplace(key, val);
                if (inserted) {
                    ++entries_count_;
                    // Update statistics
                    ++container_stats_[Index].total_inserts;
                    ++container_stats_[Index].current_size;
                    ++container_stats_[Index].value_size_distribution[value.size()];
                    
                    // Check if rehashing occurred
                    if (map.bucket_count() != bucket_count_before) {
                        ++container_stats_[Index].rehash_count;
                    }
                    
                    update_metadata_on_insert(Index, current_versions_[Index], key, height);
                }
                return inserted;
            } catch (boost::interprocess::bad_alloc const& e) {
                log_print("Error inserting into container {}: {}\n", Index, e.what());
                log_print("Next load factor: {:.2f}\n", next_load_factor<Index>());
                log_print("Current size: {}\n", container<Index>().size());
                log_print("Current bucket count: {}\n", container<Index>().bucket_count());
                log_print("Current load factor: {:.2f}\n", container<Index>().load_factor());
                log_print("Current max load factor: {:.2f}\n", container<Index>().max_load_factor());
                
                // Try to get memory info
                if (segments_[Index]) {
                    try {
                        size_t free_memory = segments_[Index]->get_free_memory();
                        log_print("Free memory in segment: {}\n", free_memory);
                    } catch (...) {
                        log_print("Cannot determine free memory in segment\n");
                    }
                }
                
                new_version<Index>();

                log_print("Retrying insert into container {} ...\n", Index);
                auto& new_map = container<Index>();
                log_print("Current size: {}\n", new_map.size());
                log_print("Current bucket count: {}\n", new_map.bucket_count());
                log_print("Current load factor: {:.2f}\n", new_map.load_factor());
                log_print("Current max load factor: {:.2f}\n", new_map.max_load_factor());                
            }       
            --max_retries;            
        }
        log_print("Failed to insert after 3 retries\n");
        throw boost::interprocess::bad_alloc();
    }
    
    template <size_t Index>
    bool should_rotate() const {
        auto const& map = container<Index>();
        if (map.bucket_count() == 0) return false;
        
        // Use the safer check
        return !can_insert_safely<Index>();
    }
    
    template <size_t Index>
        requires (Index < IdxN)
    float next_load_factor() const {
        auto& map = container<Index>();
        if (map.bucket_count() == 0) {
            return 0.0f;
        }
        return float(map.size() + 1) / float(map.bucket_count());
    }

    // Find in latest version
    std::optional<std::vector<uint8_t>> find_in_latest_version(utxo_key_t const& key, uint32_t height) {
        std::optional<std::vector<uint8_t>> result;
        
        for_each_index<IdxN>([&](auto I) {
            if (!result) {
                auto& map = container<I>();
                if (auto it = map.find(key); it != map.end()) {
                    search_stats_.add_record(height, it->second.block_height, 0, false, true, 'f');
                    auto data = it->second.get_data();
                    result = std::vector<uint8_t>(data.begin(), data.end());
                }
            }
        });
        
        return result;
    }
    
    // Find in previous versions
    std::optional<std::vector<uint8_t>> find_in_previous_versions(utxo_key_t const& key, uint32_t height) {
        std::optional<std::vector<uint8_t>> result;
        
        for_each_index<IdxN>([&](auto I) {
            if (!result) {
                result = find_in_prev_versions<I>(key, height);
            }
        });
        
        if (!result) {
            search_stats_.add_record(height, 0, 1, false, false, 'f');
        }
        
        return result;
    }
    
    template <size_t Index>
    std::optional<std::vector<uint8_t>> find_in_prev_versions(utxo_key_t const& key, uint32_t height) {
        for (size_t v = current_versions_[Index]; v-- > 0;) {
            // Check metadata
            if (file_metadata_[Index].size() > v && !file_metadata_[Index][v].key_in_range(key)) {
                continue;
            }
            
            auto [map, cache_hit] = file_cache_.get_or_open_file<Index>(Index, v);
            
            if (auto it = map.find(key); it != map.end()) {
                size_t depth = current_versions_[Index] - v;
                search_stats_.add_record(height, it->second.block_height, depth, cache_hit, true, 'f');
                auto data = it->second.get_data();
                return std::vector<uint8_t>(data.begin(), data.end());
            }
        }
        
        return std::nullopt;
    }
    
    // Erase in latest version
    size_t erase_in_latest_version(utxo_key_t const& key, uint32_t height) {
        size_t result = 0;
        
        for_each_index<IdxN>([&](auto I) {
            if (result == 0) {
                auto& map = container<I>();
                if (auto it = map.find(key); it != map.end()) {
                    // Track UTXO lifetime
                    uint32_t age = height - it->second.block_height;
                    ++lifetime_stats_.age_distribution[age];
                    lifetime_stats_.max_age = std::max(lifetime_stats_.max_age, age);
                    ++lifetime_stats_.total_spent;
                    
                    // Update average age incrementally
                    lifetime_stats_.average_age = 
                        (lifetime_stats_.average_age * (lifetime_stats_.total_spent - 1) + age) 
                        / lifetime_stats_.total_spent;
                    
                    search_stats_.add_record(height, it->second.block_height, 0, false, true, 'e');
                    map.erase(it);
                    
                    // Update container stats
                    --container_stats_[I].current_size;
                    ++container_stats_[I].total_deletes;
                    
                    result = 1;
                }
            }
        });
        
        return result;
    }    

    // Erase from cached files only
    size_t erase_from_cached_files_only(utxo_key_t const& key, uint32_t height, size_t& search_depth) {
        size_t result = 0;
        
        auto cached_files = file_cache_.get_cached_files();
        
        for (auto const& [container_index, version] : cached_files) {
            ++search_depth;
            
            // Dispatch to correct container type
            auto process_file = [&]<size_t Index>(std::integral_constant<size_t, Index>) {
                if (file_cache_.is_cached(container_index, version)) {
                    try {
                        auto [map, cache_hit] = file_cache_.get_or_open_file<Index>(container_index, version);
                        
                        if (auto it = map.find(key); it != map.end()) {
                            // Track UTXO lifetime
                            uint32_t age = height - it->second.block_height;
                            ++lifetime_stats_.age_distribution[age];
                            lifetime_stats_.max_age = std::max(lifetime_stats_.max_age, age);
                            ++lifetime_stats_.total_spent;
                            lifetime_stats_.average_age = 
                                (lifetime_stats_.average_age * (lifetime_stats_.total_spent - 1) + age) 
                                / lifetime_stats_.total_spent;
                            
                            search_stats_.add_record(height, it->second.block_height, 
                                                current_versions_[Index] - version, cache_hit, true, 'e');
                            map.erase(it);
                            
                            --container_stats_[Index].current_size;
                            ++container_stats_[Index].total_deletes;
                            
                            update_metadata_on_delete(Index, version);
                            result = 1;
                        }
                    } catch (std::exception const& e) {
                        log_print("Error accessing cached file ({}, v{}): {}\n", 
                                container_index, version, e.what());
                    }
                }
            };
            
            switch (container_index) {
                case 0: process_file(std::integral_constant<size_t, 0>{}); break;
                case 1: process_file(std::integral_constant<size_t, 1>{}); break;
                case 2: process_file(std::integral_constant<size_t, 2>{}); break;
                case 3: process_file(std::integral_constant<size_t, 3>{}); break;
            }
            
            if (result > 0) break;
        }
        
        return result;
    }    

    // Deferred deletion helpers
    void add_to_deferred_deletions(utxo_key_t const& key, uint32_t height) {
        auto [it, inserted] = deferred_deletions_.emplace(key, height);
        if (inserted) {
            ++deferred_stats_.total_deferred;
            deferred_stats_.max_queue_size = std::max(deferred_stats_.max_queue_size, deferred_deletions_.size());
            
            // Estimate which container based on typical sizes (you might want to store actual size)
            // For now, we'll increment all containers' deferred counts
            for (size_t i = 0; i < IdxN; ++i) {
                ++container_stats_[i].deferred_deletes;
            }
        }
    }
    
    template <size_t Index>
        requires (Index < IdxN)
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

                // Intenta crear el mapa con la cantidad de buckets "mid"
                using temp_map_t = utxo_map<container_sizes[Index]>;
                auto* map = segment.find_or_construct<temp_map_t>("temp_map")(
                    mid,
                    key_hash{},
                    key_equal{},
                    segment.get_allocator<std::pair<utxo_key_t const, utxo_value<container_sizes[Index]>>>()
                );

                // Si llega aquí, la cantidad de buckets es válida
                best_buckets = mid;
                left = mid + 1;
                log_print("Buckets {} successful. Increasing range...\n", mid);
            } catch (boost::interprocess::bad_alloc const& e) {
                log_print("Failed with {} buckets: {}\n", mid, e.what());
                right = mid - 1;
            }

            // Eliminar el archivo temporal en cada intento
            std::filesystem::remove(temp_file);
        }

        log_print("Optimal number of buckets: {}\n", best_buckets);
        return best_buckets;
    }

    // Process deferred deletions for a specific file (container index and version)
    size_t process_deferred_deletions_in_file(size_t container_index, size_t version, bool is_cached) {
        if (deferred_deletions_.empty()) return 0;
        
        size_t successful_deletions = 0;
        
        auto process_with_container = [&]<size_t Index>(std::integral_constant<size_t, Index>) -> size_t {
            try {
                auto [map, cache_hit] = file_cache_.get_or_open_file<Index>(container_index, version);
                
                auto it = deferred_deletions_.begin();
                while (it != deferred_deletions_.end()) {
                    auto erased_count = map.erase(it->key);
                    if (erased_count > 0) {
                        update_metadata_on_delete(Index, version);
                        size_t depth = current_versions_[Index] - version;
                        
                        // Track depth of deferred deletions
                        ++deferred_stats_.deletions_by_depth[depth];
                        
                        search_stats_.add_record(it->height, 0, depth, cache_hit, true, 'e');
                        
                        // Update container stats
                        --container_stats_[Index].deferred_deletes;
                        
                        it = deferred_deletions_.erase(it);
                        ++successful_deletions;
                    } else {
                        ++it;
                    }
                }
                
                if (successful_deletions > 0) {
                    log_print("Processed {} deletions from {}({}, v{}) - {} remaining\n", 
                            successful_deletions, is_cached ? "cached " : "", 
                            container_index, version, deferred_deletions_.size());
                }
                
                return successful_deletions;
                
            } catch (std::exception const& e) {
                log_print("Error processing file ({}, v{}): {}\n", container_index, version, e.what());
                return 0;
            }
        };
        
        // Dispatch to the correct container type
        switch (container_index) {
            case 0: return process_with_container(std::integral_constant<size_t, 0>{});
            case 1: return process_with_container(std::integral_constant<size_t, 1>{});
            case 2: return process_with_container(std::integral_constant<size_t, 2>{});
            case 3: return process_with_container(std::integral_constant<size_t, 3>{});
            default: return 0;
        }
    }

    // Process deferred deletions for a specific OP_RETURN file version
    size_t process_deferred_deletions_in_op_return_file(size_t version) {
        if (deferred_deletions_.empty()) return 0;
        
        auto op_return_file = fmt::format(op_return_file_format, db_path_.string(), version);
        size_t successful_deletions = 0;
        
        try {
            // Open the old OP_RETURN file
            auto old_segment = std::make_unique<bip::managed_mapped_file>(
                bip::open_read_only, op_return_file.c_str());
            
            auto* old_set = old_segment->find<op_return_set_t>("OPReturnSet").first;
            if (!old_set) {
                log_print("Could not find OPReturnSet in file v{}\n", version);
                return 0;
            }
            
            // Process deferred deletions against this old OP_RETURN set
            auto it = deferred_deletions_.begin();
            while (it != deferred_deletions_.end()) {
                if (old_set->count(it->key)) {
                    // Found the key in this old OP_RETURN version
                    // Since we can't modify read-only files, we'll just mark it as processed
                    log_print("Found OP_RETURN key in old version v{}: ", version);
                    print_key(it->key);
                    
                    // Update statistics
                    ++op_return_stats_.total_deletes;
                    ++successful_deletions;
                    
                    // Remove from deferred deletions
                    it = deferred_deletions_.erase(it);
                } else {
                    ++it;
                }
            }
            
            if (successful_deletions > 0) {
                log_print("Processed {} OP_RETURN deletions from old version v{} - {} remaining\n", 
                         successful_deletions, version, deferred_deletions_.size());
            }
            
        } catch (std::exception const& e) {
            log_print("Error processing OP_RETURN file v{}: {}\n", version, e.what());
            return 0;
        }
        
        return successful_deletions;
    }

    // File management
    template <size_t Index>
    void open_or_create_container(size_t version) {
        auto file_name = fmt::format(data_file_format, db_path_.string(), Index, version);
        
        segments_[Index] = std::make_unique<bip::managed_mapped_file>(
            bip::open_or_create, file_name.c_str(), file_sizes[Index]);
        
        auto* segment = segments_[Index].get();
        containers_[Index] = segment->find_or_construct<utxo_map<container_sizes[Index]>>("db_map")(
            min_buckets_ok_[Index],
            key_hash{},
            key_equal{},
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
        // Reset metadata for the new version or load if it exists (though typically new means empty)
        if (file_metadata_[Index].size() <= current_versions_[Index]) {
            file_metadata_[Index].resize(current_versions_[Index] + 1);
        }
        file_metadata_[Index][current_versions_[Index]] = file_metadata{}; // Reset for new version
        
        open_or_create_container<Index>(current_versions_[Index]);
        log_print("Container {} rotated to version {}\n", Index, current_versions_[Index]);
        // Note: OP_RETURN set does not version in this design, it's a single persistent set.
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
        // ... save implementation
    }
    
    void load_metadata_from_disk(size_t index, size_t version) {
        auto metadata_file = fmt::format("{}/meta_{}_{:05}.dat", db_path_.string(), index, version);
        // ... load implementation
        // TODO: Implement actual loading from disk for file_metadata_[index][version]
    }

    // OP_RETURN set file management
    void open_or_create_op_return_set() {
        // Find the latest version from existing files
        op_return_current_version_ = find_latest_op_return_version();
        
        auto file_path = fmt::format(op_return_file_format, db_path_.string(), op_return_current_version_);
        bool new_file = !fs::exists(file_path);

        try {
            op_return_segment_ = std::make_unique<bip::managed_mapped_file>(
                bip::open_or_create, file_path.c_str(), op_return_file_size);

            if (new_file) {
                op_return_set_ = op_return_segment_->construct<op_return_set_t>("OPReturnSet")(
                    op_return_segment_->get_segment_manager());
                
                // Initialize metadata for new version
                if (op_return_metadata_.size() <= op_return_current_version_) {
                    op_return_metadata_.resize(op_return_current_version_ + 1);
                }
                op_return_metadata_[op_return_current_version_] = file_metadata{};
                
                log_print("Created new OP_RETURN set file v{}: {}\n", op_return_current_version_, file_path);
            } else {
                op_return_set_ = op_return_segment_->find_or_construct<op_return_set_t>("OPReturnSet")(
                    op_return_segment_->get_segment_manager());
                
                // Load metadata for all versions
                for (size_t v = 0; v <= op_return_current_version_; ++v) {
                    load_op_return_metadata(v);
                }
                
                log_print("Opened existing OP_RETURN set file v{}: {}\n", op_return_current_version_, file_path);
            }
            
            op_return_stats_.current_size = op_return_set_ ? op_return_set_->size() : 0;
            
        } catch (bip::interprocess_exception const& e) {
            log_print("ERROR: Failed to open or create OP_RETURN set file {}: {}\n", file_path, e.what());
            op_return_segment_.reset();
            op_return_set_ = nullptr;
            // Potentially rethrow or handle more gracefully
            throw;
        }
    }

    void close_op_return_set() {
        if (op_return_segment_) {
            // save_op_return_metadata(); // You'll need to implement this
            // bip::managed_mapped_file::flush(*op_return_segment_); // Optional: ensure data is written
            op_return_segment_.reset(); // This will unmap and close the file
            op_return_set_ = nullptr;
            log_print("Closed OP_RETURN set file.\n");
        }
    }
    
    // Helper functions for OP_RETURN version management
    size_t find_latest_op_return_version() {
        size_t latest_version = 0;
        
        // Look for existing OP_RETURN files with version numbers
        for (size_t v = 0; v < 10000; ++v) { // Reasonable upper limit
            auto file_path = fmt::format(op_return_file_format, db_path_.string(), v);
            if (fs::exists(file_path)) {
                latest_version = v;
            } else if (v > latest_version + 100) {
                // If we haven't found a file in 100 iterations past the latest, stop
                break;
            }
        }
        
        return latest_version;
    }
    
    void load_op_return_metadata(size_t version) {
        // Ensure metadata vector is large enough
        if (op_return_metadata_.size() <= version) {
            op_return_metadata_.resize(version + 1);
        }
        
        auto metadata_file = fmt::format("{}/op_return_meta_v{:05}.dat", db_path_.string(), version);
        
        // For now, initialize empty metadata (implement actual loading later if needed)
        op_return_metadata_[version] = file_metadata{};
        op_return_metadata_[version].container_index = SIZE_MAX; // Special marker for OP_RETURN
        op_return_metadata_[version].version = version;
    }
    
    void save_op_return_metadata(size_t version) {
        if (version < op_return_metadata_.size()) {
            auto metadata_file = fmt::format("{}/op_return_meta_v{:05}.dat", db_path_.string(), version);
            // TODO: Implement actual saving to disk for op_return_metadata_[version]
            // For now, metadata is stored in memory
        }
    }
    
    void update_op_return_metadata_on_insert(size_t version, utxo_key_t const& key) {
        if (op_return_metadata_.size() <= version) {
            op_return_metadata_.resize(version + 1);
        }
        
        auto& metadata = op_return_metadata_[version];
        if (metadata.entry_count == 0) {
            metadata.min_key = metadata.max_key = key;
        } else {
            if (key < metadata.min_key) metadata.min_key = key;
            if (key > metadata.max_key) metadata.max_key = key;
        }
        metadata.entry_count++;
    }
    
    void update_op_return_metadata_on_delete(size_t version) {
        if (version < op_return_metadata_.size() && op_return_metadata_[version].entry_count > 0) {
            op_return_metadata_[version].entry_count--;
        }
    }
    
    // Check if OP_RETURN set needs rotation
    bool op_return_needs_rotation() const {
        if (!op_return_set_) return false;
        
        // Check if the set is getting close to capacity
        // We'll use a heuristic based on load factor and available memory
        try {
            if (op_return_segment_) {
                size_t free_memory = op_return_segment_->get_free_memory();
                size_t current_size = op_return_set_->size();
                
                // Rotate if less than 10% memory is free or if we hit a size threshold
                double memory_usage_ratio = 1.0 - (double(free_memory) / double(op_return_file_size));
                
                if (memory_usage_ratio > 0.9 || current_size > 1000000) { // 1M entries threshold
                    return true;
                }
            }
        } catch (...) {
            // If we can't determine memory usage, be conservative and rotate
            return op_return_set_->size() > 500000; // 500K entries fallback
        }
        
        return false;
    }
    
    // Rotate OP_RETURN set to new version
    void rotate_op_return_set() {
        log_print("Rotating OP_RETURN set from version {} to {}\n", 
                 op_return_current_version_, op_return_current_version_ + 1);
        
        // Simply close current set (don't cache it)
        close_op_return_set();
        
        // Increment version and create new file
        ++op_return_current_version_;
        
        auto file_path = fmt::format(op_return_file_format, db_path_.string(), op_return_current_version_);
        
        try {
            op_return_segment_ = std::make_unique<bip::managed_mapped_file>(
                bip::create_only, file_path.c_str(), op_return_file_size);
            
            op_return_set_ = op_return_segment_->construct<op_return_set_t>("OPReturnSet")(
                op_return_segment_->get_segment_manager());
            
            // Initialize metadata for new version
            if (op_return_metadata_.size() <= op_return_current_version_) {
                op_return_metadata_.resize(op_return_current_version_ + 1);
            }
            op_return_metadata_[op_return_current_version_] = file_metadata{};
            op_return_metadata_[op_return_current_version_].container_index = SIZE_MAX;
            op_return_metadata_[op_return_current_version_].version = op_return_current_version_;
            
            // Reset current size since we have a new empty set
            op_return_stats_.current_size = 0;
            
            log_print("Created new OP_RETURN set file v{}: {}\n", op_return_current_version_, file_path);
            
        } catch (bip::interprocess_exception const& e) {
            log_print("ERROR: Failed to create new OP_RETURN set file v{}: {}\n", 
                     op_return_current_version_, e.what());
            throw;
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
    
    // Helper to check if container can accommodate new insertions
    template <size_t Index>
    bool can_insert_safely() const {
        auto const& map = container<Index>();
        
        // Check load factor
        if (map.bucket_count() > 0) {
            float next_load = float(map.size() + 1) / float(map.bucket_count());
            if (next_load >= map.max_load_factor() * 0.95f) {
                return false;
            }
        }
        
        // Check available memory
        if (segments_[Index]) {
            try {
                size_t free_memory = segments_[Index]->get_free_memory();
                size_t entry_size = sizeof(typename utxo_map<container_sizes[Index]>::value_type);
                size_t buffer_size = entry_size * 10; // Safety buffer for 10 entries
                
                return free_memory > buffer_size;
            } catch (...) {
                return false; // If we can't check, assume not safe
            }
        }
        
        return true; // If no segment, assume it's safe (shouldn't happen)
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
            auto file_name = fmt::format(data_file_format, db_path_.string(), index, v);
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