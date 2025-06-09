// like interprocess_multiple_v0.hpp but using bloom filters

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
// #include <unordered_map>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <regex>
#include <fstream>
#include <set>
#include <span>

#define FMT_HEADER_ONLY 1
#include <fmt/core.h>

#include <boost/interprocess/managed_mapped_file.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/container_hash/hash.hpp>

#include "common_db.hpp"
#include "log.hpp"


#include <boost/interprocess/managed_mapped_file.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/container_hash/hash.hpp>
#include <array>
#include <vector>
#include <iostream>
#include <filesystem>
#include <cstdint>
#include <type_traits>


#include <utility>  // std::index_sequence, std::make_index_sequence

// ---------------------------------------------------

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

// ---------------------------------------------------

namespace bip = boost::interprocess;

constexpr size_t utxo_key_size = 36;   // 32 bytes hash + 4 bytes index

// Bucket configuration (modifiable)
constexpr std::array<size_t, 4> container_sizes = {44, 128, 512, 10011};
constexpr std::array<size_t, 4> file_sizes = {
    400_mib,
    400_mib,
    200_mib,
    100_mib
};
// constexpr std::array<size_t, 1> container_sizes = {44};
// constexpr std::array<size_t, 1> file_sizes = {
//     400_mib
// };

using utxo_key_t = std::array<std::uint8_t, utxo_key_size>;

// Select appropriate uint type for size
constexpr auto smallest_uint_type(size_t size) {
    return size <= 0xFF   ? std::uint8_t{}  :
           size <= 0xFFFF ? std::uint16_t{} :
           size <= 0xFFFFFFFF ? std::uint32_t{} :
           std::uint64_t{};
}

template <size_t Size>
using utxo_val_t = std::tuple<uint32_t, decltype(smallest_uint_type(Size)), std::array<std::uint8_t, Size>>;

// Individual search record for detailed statistics
struct search_record {
    uint32_t access_height;      // Block height when the search/erase occurred
    uint32_t insertion_height;   // Block height when the element was inserted (0 if not found)
    uint32_t depth;              // 0 = current version, 1+ = versions back, cache info in is_cache_hit
    bool is_cache_hit;           // true if found in cache (only relevant when depth > 0)
    bool found;                  // true if the key was found, false if not found
    char operation;              // 'f' for find, 'e' for erase
    
    search_record(uint32_t access_h, uint32_t insert_h, uint32_t d, bool cache, bool f, char op) 
        : access_height(access_h), insertion_height(insert_h), depth(d), is_cache_hit(cache), found(f), operation(op) {}
        
    // Calculate age of UTXO when accessed (only meaningful if found)
    uint32_t get_utxo_age() const {
        return found && access_height >= insertion_height ? access_height - insertion_height : 0;
    }
};

// Aggregated statistics result structure
struct search_statistics_result {
    size_t total_operations = 0;
    size_t found_operations = 0;
    size_t not_found_operations = 0;
    size_t find_operations = 0;
    size_t erase_operations = 0;
    size_t current_version_hits = 0;
    size_t prev_version_accesses = 0;
    size_t cache_hits = 0;
    size_t cache_accesses = 0;
    
    double avg_depth = 0.0;
    double avg_utxo_age = 0.0;
    double cache_hit_rate = 0.0;
    double current_hit_rate = 0.0;
    double found_rate = 0.0;
    
    uint32_t max_depth = 0;
    uint32_t min_utxo_age = UINT32_MAX;
    uint32_t max_utxo_age = 0;
};

// Statistics for tracking search operations
struct search_stats {
    std::vector<search_record> search_records;
    
    void reset() {
        search_records.clear();
    }
    
    // Get all statistics for a height range in a single pass - much more efficient
    search_statistics_result get_statistics_for_access_height_range(uint32_t min_height, uint32_t max_height) const {
        search_statistics_result result;
        
        size_t total_depth = 0;
        size_t total_age = 0;
        size_t found_with_age = 0;
        
        for (const auto& record : search_records) {
            if (record.access_height >= min_height && record.access_height <= max_height) {
                result.total_operations++;
                total_depth += record.depth;
                
                if (record.found) {
                    result.found_operations++;
                    if (record.depth == 0) {
                        result.current_version_hits++;
                    }
                    
                    uint32_t utxo_age = record.get_utxo_age();
                    if (utxo_age > 0) {
                        total_age += utxo_age;
                        found_with_age++;
                        result.min_utxo_age = std::min(result.min_utxo_age, utxo_age);
                        result.max_utxo_age = std::max(result.max_utxo_age, utxo_age);
                    }
                } else {
                    result.not_found_operations++;
                }
                
                if (record.operation == 'f') {
                    result.find_operations++;
                } else if (record.operation == 'e') {
                    result.erase_operations++;
                }
                
                if (record.depth > 0) {
                    result.prev_version_accesses++;
                    result.cache_accesses++;
                    if (record.is_cache_hit) {
                        result.cache_hits++;
                    }
                }
                
                result.max_depth = std::max(result.max_depth, record.depth);
            }
        }
        
        // Calculate rates and averages
        if (result.total_operations > 0) {
            result.avg_depth = double(total_depth) / double(result.total_operations);
            result.current_hit_rate = double(result.current_version_hits) / double(result.total_operations);
            result.found_rate = double(result.found_operations) / double(result.total_operations);
        }
        
        if (found_with_age > 0) {
            result.avg_utxo_age = double(total_age) / double(found_with_age);
        }
        
        if (result.cache_accesses > 0) {
            result.cache_hit_rate = double(result.cache_hits) / double(result.cache_accesses);
        }
        
        // Handle case where no UTXOs with age were found
        if (result.min_utxo_age == UINT32_MAX) {
            result.min_utxo_age = 0;
        }
        
        return result;
    }
    
    // Get statistics for all records
    search_statistics_result get_all_statistics() const {
        log_print("search_stats: get_all_statistics called, found {} records\n", search_records.size());
        if (search_records.empty()) {
            return search_statistics_result{};
        }
        
        uint32_t min_height = UINT32_MAX;
        uint32_t max_height = 0;
        
        for (const auto& record : search_records) {
            min_height = std::min(min_height, record.access_height);
            max_height = std::max(max_height, record.access_height);
        }
        
        return get_statistics_for_access_height_range(min_height, max_height);
    }
};

using segment_manager_t = bip::managed_mapped_file::segment_manager;
using key_hash = boost::hash<utxo_key_t>;
using key_equal = std::equal_to<utxo_key_t>;

// File metadata structure for range filtering
struct file_metadata {
    uint32_t min_block_height = UINT32_MAX;
    uint32_t max_block_height = 0;
    utxo_key_t min_key = {
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff
    };
    utxo_key_t max_key = {};
    size_t entry_count = 0;
    size_t container_index = 0;
    size_t version = 0;
    
    // // Initialize with extreme values
    // file_metadata() {
    //     min_key.fill(0xFF);  // Maximum possible key
    //     max_key.fill(0x00);  // Minimum possible key
    // }
    
    // Check if a key might be in this file's range
    bool key_in_range(utxo_key_t const& key) const {
        if (entry_count == 0) return false;
        return key >= min_key && key <= max_key;
    }
    
    // Check if a block height might be in this file's range
    bool block_in_range(uint32_t height) const {
        if (entry_count == 0) return false;
        return height >= min_block_height && height <= max_block_height;
    }
    
    // Update metadata when inserting a new entry
    void update_on_insert(utxo_key_t const& key, uint32_t height) {
        if (entry_count == 0) {
            // First entry
            min_key = key;
            max_key = key;
            min_block_height = height;
            max_block_height = height;
        } else {
            // Update ranges
            if (key < min_key) min_key = key;
            if (key > max_key) max_key = key;
            if (height < min_block_height) min_block_height = height;
            if (height > max_block_height) max_block_height = height;
        }
        entry_count++;
    }
    
    // Update metadata when deleting an entry
    void update_on_delete() {
        if (entry_count > 0) {
            entry_count--;
        }
        // Note: We can't easily update min/max on delete without scanning the entire file
        // For now, we'll keep the existing ranges (they become upper bounds)
        // In a future optimization, we could periodically rebuild metadata
    }
};

template <size_t Index>
    requires (Index < container_sizes.size())
using map_t = boost::unordered_flat_map<
    utxo_key_t,
    utxo_val_t<container_sizes[Index]>,
    key_hash,
    key_equal,
    bip::allocator<
        std::pair<
            utxo_key_t const, 
            utxo_val_t<container_sizes[Index]>
        >,
        segment_manager_t
    >
>;

// Structure to hold comprehensive file cache statistics
struct file_cache_stats {
    size_t cache_size = 0;              // Current number of files in cache
    size_t max_cache_size = 0;          // Maximum cache capacity
    size_t total_gets = 0;              // Total get operations
    size_t total_hits = 0;              // Total cache hits
    size_t total_evictions = 0;         // Total evictions
    float hit_rate = 0.0f;              // Cache hit rate (hits/gets)
    size_t pinned_files = 0;            // Number of currently pinned files
    size_t tracked_files = 0;           // Number of files being tracked for access frequency
    
    // Per-file statistics
    struct file_info {
        std::string path;
        size_t access_count = 0;
        bool is_pinned = false;
        bool is_in_cache = false;
        std::chrono::steady_clock::time_point last_used;
    };
    
    std::vector<file_info> files;       // Information about all tracked files
};

class file_cache {
    struct cached_file {
        std::unique_ptr<bip::managed_mapped_file> segment;
        void* map_ptr;
        std::chrono::steady_clock::time_point last_used;
        size_t access_count = 0;  // Track frequency of access
        bool is_pinned = false;   // Prevent eviction of critical files
    };
    
    boost::unordered_flat_map<std::string, cached_file> cache_;
    size_t max_cached_files_ = 2; // Increased from 20 to 50
    size_t min_pinned_files_ = 1; // Keep at least this many files permanently
    
    size_t gets_ = 0;
    size_t hits_ = 0;
    size_t evictions_ = 0;
    
    // Track access patterns to identify hot files
    boost::unordered_flat_map<std::string, size_t> access_frequency_;
    
public:
    float get_hit_rate() const {
        return gets_ > 0 ? float(hits_) / float(gets_) : 0.0f;
    }
    
    size_t get_eviction_count() const {
        return evictions_;
    }
    
    void set_cache_size(size_t new_size) {
        max_cached_files_ = new_size;
    }
    
    // Get comprehensive cache statistics
    file_cache_stats get_comprehensive_stats() const {
        file_cache_stats stats;
        
        // Basic statistics
        stats.cache_size = cache_.size();
        stats.max_cache_size = max_cached_files_;
        stats.total_gets = gets_;
        stats.total_hits = hits_;
        stats.total_evictions = evictions_;
        stats.hit_rate = get_hit_rate();
        stats.tracked_files = access_frequency_.size();
        
        // Count pinned files
        stats.pinned_files = 0;
        for (const auto& [path, file] : cache_) {
            if (file.is_pinned) {
                ++stats.pinned_files;
            }
        }
        
        // Collect detailed file information
        stats.files.reserve(access_frequency_.size());
        for (const auto& [path, frequency] : access_frequency_) {
            file_cache_stats::file_info info;
            info.path = path;
            info.access_count = frequency;
            
            // Check if file is currently in cache
            auto cache_it = cache_.find(path);
            if (cache_it != cache_.end()) {
                info.is_in_cache = true;
                info.is_pinned = cache_it->second.is_pinned;
                info.last_used = cache_it->second.last_used;
            } else {
                info.is_in_cache = false;
                info.is_pinned = false;
            }
            
            stats.files.push_back(std::move(info));
        }
        
        // Sort files by access count (most accessed first)
        std::sort(stats.files.begin(), stats.files.end(),
            [](const auto& a, const auto& b) { 
                return a.access_count > b.access_count; 
            });
        
        return stats;
    }
    
    // Pin frequently accessed files to prevent eviction
    void pin_hot_files() {
        if (access_frequency_.empty()) return;
        
        // Sort files by access frequency
        std::vector<std::pair<std::string, size_t>> sorted_files(
            access_frequency_.begin(), access_frequency_.end());
        std::sort(sorted_files.begin(), sorted_files.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Pin the most frequently accessed files
        size_t pinned = 0;
        for (const auto& [file_path, freq] : sorted_files) {
            if (pinned >= min_pinned_files_) break;
            auto it = cache_.find(file_path);
            if (it != cache_.end()) {
                it->second.is_pinned = true;
                ++pinned;
            }
        }
        
        log_print("file_cache: pinned {} hot files\n", pinned);
    }

    template <size_t Index>
        requires (Index < container_sizes.size())
    std::pair<map_t<Index>&, bool> get_or_open_file(std::string const& file_path) {
        ++gets_;
        auto const now = std::chrono::steady_clock::now();
        
        // Track access frequency
        ++access_frequency_[file_path];
        
        // Check if already in cache
        auto const it = cache_.find(file_path);
        if (it != cache_.end()) {
            it->second.last_used = now;
            ++it->second.access_count;
            ++hits_;
            return {*static_cast<map_t<Index>*>(it->second.map_ptr), true};
        }
        
        // Try to evict if cache is full, but be smarter about it
        if (cache_.size() >= max_cached_files_) {
            // First try to evict non-pinned files only
            auto lru = cache_.end();
            auto oldest_time = std::chrono::steady_clock::time_point::max();
            
            for (auto iter = cache_.begin(); iter != cache_.end(); ++iter) {
                if (!iter->second.is_pinned && iter->second.last_used < oldest_time) {
                    oldest_time = iter->second.last_used;
                    lru = iter;
                }
            }
            
            // If no non-pinned files found, evict the oldest pinned file
            if (lru == cache_.end()) {
                lru = std::min_element(cache_.begin(), cache_.end(),
                    [](const auto& a, const auto& b) {
                        return a.second.last_used < b.second.last_used;
                    });
            }
            
            if (lru != cache_.end()) {
                log_print("file_cache: evicting {} (pinned: {}, access_count: {})\n", 
                    lru->first, lru->second.is_pinned, lru->second.access_count);
                cache_.erase(lru);
                ++evictions_;
            }
        }
        
        // Open new file
        try {
            auto segment = std::make_unique<bip::managed_mapped_file>(
                bip::open_only, file_path.c_str());
                
            auto* map = segment->find<map_t<Index>>("db_map").first;
            if (!map) {
                log_print("file_cache: ERROR - failed to find db_map in {}\n", file_path);
                throw std::runtime_error("Failed to find db_map in file");
            }
            
            // Store in cache
            cached_file new_file{std::move(segment), map, now, 1, false};
            cache_[file_path] = std::move(new_file);
            return {*static_cast<map_t<Index>*>(map), false};
            
        } catch (const std::exception& e) {
            log_print("file_cache: ERROR opening {}: {}\n", file_path, e.what());
            throw;
        }
    }
    
    // Try to add an existing segment to cache (used when moving latest file to cache)
    template <size_t Index>
        requires (Index < container_sizes.size())
    std::pair<map_t<Index>&, bool> try_add_to_cache(const std::string& file_path, 
                                                     std::unique_ptr<bip::managed_mapped_file> segment,
                                                     void* map_ptr) {
        auto const now = std::chrono::steady_clock::now();
        
        // Track access frequency
        ++access_frequency_[file_path];
        
        // Check if already in cache
        auto const it = cache_.find(file_path);
        if (it != cache_.end()) {
            it->second.last_used = now;
            ++it->second.access_count;
            return {*static_cast<map_t<Index>*>(it->second.map_ptr), true};
        }
        
        // Try to make room if cache is full
        if (cache_.size() >= max_cached_files_) {
            // Try to evict a non-pinned file
            auto lru = cache_.end();
            auto oldest_time = std::chrono::steady_clock::time_point::max();
            
            for (auto iter = cache_.begin(); iter != cache_.end(); ++iter) {
                if (!iter->second.is_pinned && iter->second.last_used < oldest_time) {
                    oldest_time = iter->second.last_used;
                    lru = iter;
                }
            }
            
            if (lru != cache_.end()) {
                log_print("file_cache: evicting {} to make room for latest file {}\n", 
                         lru->first, file_path);
                cache_.erase(lru);
                ++evictions_;
            }
        }
        
        // Add to cache if we have room
        if (cache_.size() < max_cached_files_) {
            cached_file new_file{std::move(segment), map_ptr, now, 1, true}; // Pin the latest file
            cache_[file_path] = std::move(new_file);
            
            log_print("file_cache: added latest file {} to cache (pinned)\n", file_path);
            return {*static_cast<map_t<Index>*>(map_ptr), false};
        } else {
            // Cache is full and no evictable files, just return the map
            log_print("file_cache: cache full, cannot add latest file {}\n", file_path);
            return {*static_cast<map_t<Index>*>(map_ptr), false};
        }
    }

    // Periodic maintenance to update pinning based on access patterns
    void maintain_cache() {
        static size_t maintenance_counter = 0;
        if (++maintenance_counter % 1000 == 0) { // Every 1000 accesses
            pin_hot_files();
            
            // Log cache statistics
            log_print("file_cache: size={}, hits={}, gets={}, hit_rate={:.2f}%, evictions={}\n",
                cache_.size(), hits_, gets_, get_hit_rate() * 100.0f, evictions_);
        }
    }
};

class utxo_db {
    using span_bytes = std::span<uint8_t const>;

public:
    utxo_db() = default;

    // replace with static named ctor + private ctor
    void configure(
        std::string_view path, 
        // size_t file_size = default_file_size,
        // size_t min_buckets = default_min_buckets,
        bool remove_existing = false) 
    {
        db_path_ = path;
        // file_size_ = file_size;
        // min_buckets_ok_.fill(min_buckets);
        // min_buckets_ok_.fill(0);

        if (remove_existing && std::filesystem::exists(path)) {
            std::filesystem::remove_all(path);
        }
        std::filesystem::create_directories(path);

        // auto optimal0 = find_optimal_buckets<0>("./optimal", file_sizes[0], 7864304);
        // log_print("Optimal number of buckets for container {} and file size {}: {}\n", 0, file_sizes[0], optimal0);
        // auto optimal1 = find_optimal_buckets<1>("./optimal", file_sizes[1], 7864304);
        // log_print("Optimal number of buckets for container {} and file size {}: {}\n", 1, file_sizes[1], optimal1);
        // auto optimal2 = find_optimal_buckets<2>("./optimal", file_sizes[2], 7864304);
        // log_print("Optimal number of buckets for container {} and file size {}: {}\n", 2, file_sizes[2], optimal2);
        // auto optimal3 = find_optimal_buckets<3>("./optimal", file_sizes[3], 7864304);
        // log_print("Optimal number of buckets for container {} and file size {}: {}\n", 3, file_sizes[3], optimal3);

        for_each_index<container_sizes.size()>([&](auto I) {
            size_t latest_version = find_latest_version_from_files(I);
            open_or_create_container<I>(latest_version);
            
            // Load metadata for all existing versions
            for (size_t v = 0; v <= latest_version; ++v) {
                load_metadata_from_disk(I, v);
            }
        });        
    }
    
    void close() {
        for_each_index<container_sizes.size()>([&](auto I) {
            close_container<I>();
        });
    }

    template <size_t Index>
        requires (Index < container_sizes.size())
    map_t<Index>& container() {
        return *static_cast<map_t<Index>*>(containers_[Index]);
    }

    template <size_t Index>
        requires (Index < container_sizes.size())
    map_t<Index> const& container() const{
        return *static_cast<map_t<Index> const*>(containers_[Index]);
    }

    template <size_t... Is>
    bool insert_impl(size_t index, utxo_key_t const& key, span_bytes value, uint32_t height, std::index_sequence<Is...>) {
        bool result = false;
        auto try_insert = [&](auto I) {
            if (index == I) {
                result = insert_in_index<I>(key, value, height);
                return true;
            }
            return false;
        };
        (try_insert(std::integral_constant<size_t, Is>{}) || ...);
        return result;
    }

    bool insert(utxo_key_t const& key, span_bytes value, uint32_t height) {
        constexpr auto N = container_sizes.size();
        size_t index = get_index_from_size(value.size());
        if (index >= N) {
            throw std::out_of_range("Invalid index");
        }
        return insert_impl(index, key, value, height, std::make_index_sequence<N>{});
    }

    template <size_t... Is>
    size_t erase_prev_versions_impl(utxo_key_t const& key, uint32_t height, std::index_sequence<Is...>) {
        size_t res = 0;

        // Lambda para evaluar cada índice
        auto try_erase = [&](auto I) -> bool {
            res = erase_x_in_prev_versions<I>(key, height);
            return res > 0;
        };

        auto size_prev = search_stats_.search_records.size();

        // Expand fold expression con short-circuit
        (try_erase(std::integral_constant<size_t, Is>{}) || ...);

        return res;
    }

    size_t erase(utxo_key_t const& key, uint32_t height) {
        constexpr auto N = container_sizes.size();
        auto res = erase_in_latest_version(key, height);
        if (res > 0) {
            return res;
        }
        return erase_prev_versions_impl(key, height, std::make_index_sequence<N>{});
    }
        
    template <size_t... Is>
    std::optional<std::vector<uint8_t>> find_prev_versions_impl(utxo_key_t const& key, uint32_t height, std::index_sequence<Is...>) {
        std::optional<std::vector<uint8_t>> res;

        auto try_find = [&](auto I) -> bool {
            res = find_x_in_prev_versions<I>(key, height);
            return res.has_value();
        };

        (try_find(std::integral_constant<size_t, Is>{}) || ...);

        return res;
    }

    std::optional<std::vector<uint8_t>> find(utxo_key_t const& key, uint32_t height) {
        constexpr auto N = container_sizes.size();
        if (auto res = find_in_latest_version(key, height); res) return res;
        return find_prev_versions_impl(key, height, std::make_index_sequence<N>{});
    }

    template <size_t... Is>
    size_t size_current_impl(std::index_sequence<Is...>) const {
        return (container<Is>().size() + ...);
    }

    size_t size_current() const {
        constexpr auto N = container_sizes.size();
        return size_current_impl(std::make_index_sequence<N>{});
    }

    template <size_t Index>
        requires (Index < container_sizes.size())
    size_t size_container() const {
        return container<Index>().size();
    }

    // Get search statistics
    search_stats const& get_search_stats() const {
        return search_stats_;
    }

    // Reset search statistics
    void reset_search_stats() {
        search_stats_.reset();
    }

    // File cache configuration methods
    void configure_cache_size(size_t cache_size) {
        file_cache_.set_cache_size(cache_size);
        log_print("file_cache: cache size configured to {} files\n", cache_size);
    }
    
    float get_cache_hit_rate() const {
        return file_cache_.get_hit_rate();
    }
    
    size_t get_cache_eviction_count() const {
        return file_cache_.get_eviction_count();
    }
    
    void force_cache_maintenance() {
        file_cache_.pin_hot_files();
        log_print("file_cache: forced cache maintenance completed\n");
    }
    
    void print_cache_stats() const {
        log_print("File Cache Statistics:\n");
        log_print("  Hit rate: {:.2f}%\n", get_cache_hit_rate() * 100.0f);
        log_print("  Evictions: {}\n", get_cache_eviction_count());
    }
    
    // Get comprehensive file cache statistics
    file_cache_stats get_file_cache_stats() const {
        return file_cache_.get_comprehensive_stats();
    }
    
    // Print detailed file cache statistics
    void print_detailed_cache_stats() const {
        auto stats = get_file_cache_stats();
        
        log_print("Comprehensive File Cache Statistics:\n");
        log_print("  Cache size: {}/{} files\n", stats.cache_size, stats.max_cache_size);
        log_print("  Hit rate: {:.2f}% ({}/{} requests)\n", 
                  stats.hit_rate * 100.0f, stats.total_hits, stats.total_gets);
        log_print("  Evictions: {}\n", stats.total_evictions);
        log_print("  Pinned files: {}\n", stats.pinned_files);
        log_print("  Tracked files: {}\n", stats.tracked_files);
        
        if (!stats.files.empty()) {
            log_print("  Top accessed files:\n");
            size_t top_files = std::min(size_t(10), stats.files.size());
            for (size_t i = 0; i < top_files; ++i) {
                const auto& file = stats.files[i];
                log_print("    {}: {} accesses{}{}\n", 
                         file.path, file.access_count,
                         file.is_in_cache ? " [cached]" : "",
                         file.is_pinned ? " [pinned]" : "");
            }
        }
    }
    
    // Metadata management methods
    void save_metadata_to_disk(size_t container_index, size_t version) {
        auto metadata_file = fmt::format("{}/meta_{}_{:05}.dat", db_path_.string(), container_index, version);
        
        try {
            std::ofstream file(metadata_file, std::ios::binary);
            if (!file) {
                log_print("WARNING: failed to create metadata file: {}\n", metadata_file);
                return;
            }
            
            if (version < file_metadata_[container_index].size()) {
                const auto& metadata = file_metadata_[container_index][version];
                file.write(reinterpret_cast<const char*>(&metadata), sizeof(file_metadata));
            }
        } catch (const std::exception& e) {
            log_print("WARNING: failed to save metadata to {}: {}\n", metadata_file, e.what());
        }
    }
    
    void load_metadata_from_disk(size_t container_index, size_t version) {
        auto metadata_file = fmt::format("{}/meta_{}_{:05}.dat", db_path_.string(), container_index, version);
        
        // Ensure the metadata vector is large enough
        if (file_metadata_[container_index].size() <= version) {
            file_metadata_[container_index].resize(version + 1);
        }
        
        try {
            std::ifstream file(metadata_file, std::ios::binary);
            if (!file) {
                // File doesn't exist, initialize with default metadata
                file_metadata_[container_index][version] = file_metadata{};
                file_metadata_[container_index][version].container_index = container_index;
                file_metadata_[container_index][version].version = version;
                return;
            }
            
            file.read(reinterpret_cast<char*>(&file_metadata_[container_index][version]), sizeof(file_metadata));
            
            if (!file) {
                log_print("WARNING: failed to read metadata from {}, using defaults\n", metadata_file);
                file_metadata_[container_index][version] = file_metadata{};
                file_metadata_[container_index][version].container_index = container_index;
                file_metadata_[container_index][version].version = version;
            }
        } catch (const std::exception& e) {
            log_print("WARNING: failed to load metadata from {}: {}, using defaults\n", metadata_file, e.what());
            file_metadata_[container_index][version] = file_metadata{};
            file_metadata_[container_index][version].container_index = container_index;
            file_metadata_[container_index][version].version = version;
        }
    }
    
    void update_metadata_on_insert(size_t container_index, size_t version, utxo_key_t const& key, uint32_t height) {
        // Ensure the metadata vector is large enough
        if (file_metadata_[container_index].size() <= version) {
            file_metadata_[container_index].resize(version + 1);
        }
        
        file_metadata_[container_index][version].update_on_insert(key, height);
    }
    
    void update_metadata_on_delete(size_t container_index, size_t version) {
        if (file_metadata_[container_index].size() > version) {
            file_metadata_[container_index][version].update_on_delete();
        }
    }
    
    // Print metadata statistics for all files
    void print_metadata_stats() const {
        log_print("File Metadata Statistics:\n");
        
        for (size_t container_index = 0; container_index < container_sizes.size(); ++container_index) {
            log_print("  Container {} (max size: {}):\n", container_index, container_sizes[container_index]);
            
            const auto& metadata_vec = file_metadata_[container_index];
            for (size_t version = 0; version < metadata_vec.size(); ++version) {
                const auto& metadata = metadata_vec[version];
                if (metadata.entry_count > 0) {
                    log_print("    Version {}: {} entries, blocks [{}-{}]\n",
                             version, metadata.entry_count, 
                             metadata.min_block_height, metadata.max_block_height);
                    
                    // Print key range in hex (first and last 4 bytes)
                    log_print("      Key range: {:02x}{:02x}{:02x}{:02x}...{:02x}{:02x}{:02x}{:02x} to "
                             "{:02x}{:02x}{:02x}{:02x}...{:02x}{:02x}{:02x}{:02x}\n",
                             metadata.min_key[0], metadata.min_key[1], metadata.min_key[2], metadata.min_key[3],
                             metadata.min_key[32], metadata.min_key[33], metadata.min_key[34], metadata.min_key[35],
                             metadata.max_key[0], metadata.max_key[1], metadata.max_key[2], metadata.max_key[3],
                             metadata.max_key[32], metadata.max_key[33], metadata.max_key[34], metadata.max_key[35]);
                }
            }
        }
    }

private:
    // static constexpr size_t default_file_size = 1024 * 1024 * 609; // 609 MiB
    // static constexpr size_t default_min_buckets = 7864304;
    static constexpr std::string_view file_format = "{}/cont_{}_v{:05}.dat";

    std::filesystem::path db_path_ = "utxo_interprocess";
    // size_t file_size_ = default_file_size;
    // size_t min_buckets_ = default_min_buckets;

    std::array<std::unique_ptr<bip::managed_mapped_file>, container_sizes.size()> segments_;
    std::array<void*, container_sizes.size()> containers_{};
    std::array<size_t, container_sizes.size()> current_versions = {};
    std::array<size_t, container_sizes.size()> min_buckets_ok_ = {
        3932159,
        1966079,
         245759,
           7679
    };

    // std::array<size_t, container_sizes.size()> min_buckets_ok_ = {
    //     3932159
    // };

    file_cache file_cache_;
    search_stats search_stats_;
    
    // Metadata storage: vector indexed by [container_index][version] = metadata
    std::array<std::vector<file_metadata>, container_sizes.size()> file_metadata_;

    template <size_t Index>
        requires (Index < container_sizes.size())
    std::optional<std::vector<uint8_t>> find_in_latest_version(utxo_key_t const& key, uint32_t height) {
        auto it = container<Index>().find(key);
        if (it != container<Index>().end()) {
            auto const insertion_height = std::get<0>(it->second);
            search_stats_.search_records.emplace_back(height, insertion_height, 0, false, true, 'f');
            auto const& val = std::get<2>(it->second);
            return std::vector<uint8_t>(val.begin(), val.end());
        }
        return std::nullopt;
    }

    template <size_t... Is>
    std::optional<std::vector<uint8_t>> find_in_latest_version_impl(utxo_key_t const& key, uint32_t height, std::index_sequence<Is...>) {
        std::optional<std::vector<uint8_t>> res;

        auto try_find = [&](auto I) -> bool {
            res = find_in_latest_version<I>(key, height);
            return res.has_value();
        };

        (try_find(std::integral_constant<size_t, Is>{}) || ...);

        return res;
    }

    std::optional<std::vector<uint8_t>> find_in_latest_version(utxo_key_t const& key, uint32_t height) {
        constexpr auto N = container_sizes.size();
        return find_in_latest_version_impl(key, height, std::make_index_sequence<N>{});
    }



    template <size_t Index>
        requires (Index < container_sizes.size())    
    std::optional<std::vector<uint8_t>> find_x_in_prev_versions(utxo_key_t const& key, uint32_t height) {
        size_t version = current_versions[Index];
        size_t versions_back = 0;

        while (version > 0) {
            --version;
            ++versions_back;

            // Check metadata first to avoid unnecessary file access
            if (file_metadata_[Index].size() > version) {
                const auto& metadata = file_metadata_[Index][version];
                if (!metadata.key_in_range(key)) {
                    // Key is definitely not in this file, skip it
                    continue;
                }
            } else {
                // Metadata not available, we have to check the file
                log_print("Warning: metadata for container {} version {} not available, checking file directly\n", Index, version);
                throw std::runtime_error("Metadata not available for version");
            }

            auto const file_name = fmt::format(file_format, db_path_.string(), Index, version);
            auto [map, was_cache_hit] = file_cache_.get_or_open_file<Index>(file_name);

            // Call maintenance periodically
            file_cache_.maintain_cache();

            auto const it = map.find(key);
            if (it == map.end()) {
                continue; // Not found in this version
            }
            auto const insertion_height = std::get<0>(it->second);
            search_stats_.search_records.emplace_back(height, insertion_height, versions_back, was_cache_hit, true, 'f');
            auto const& val = std::get<2>(it->second);
            return std::vector<uint8_t>(val.begin(), val.end());
        }
        
        // Not found even after going through all versions
        if (versions_back > 0) {
            search_stats_.search_records.emplace_back(height, 0, versions_back, false, false, 'f');
        }
        return std::nullopt;
    }

    template <size_t Index>
        requires (Index < container_sizes.size())
    float next_load_factor() const {
        auto& map = container<Index>();
        if (map.bucket_count() == 0) {
            return 0.0f;
        }
        return float(map.size() + 1) / float(map.bucket_count());
    }


    template <size_t Index>
        requires (Index < container_sizes.size())
    bool insert_in_index_safe(utxo_key_t const& key, span_bytes value, uint32_t height) {
        auto& map = container<Index>();
        if (next_load_factor<Index>() >= map.max_load_factor()) {
            log_print("Next load factor {:.2f} exceeds max load factor {:.2f} for container {}\n", 
                      next_load_factor<Index>(), map.max_load_factor(), Index);
            throw std::out_of_range("Next load factor exceeds max load factor");
        }


        size_t max_retries = 3;
        while (max_retries > 0) {
            try {
                // log_print("Before emplace in DB for Index {}: \n", Index);
                auto res = map.emplace(key, utxo_val_t<container_sizes[Index]>{
                    height,
                    value.size(), 
                    {}
                });
                // log_print("After emplace in DB for Index {}: \n", Index);
                if (res.second) {
                    // insert took place
                    // std::copy(value.begin(), value.end(), res.first->second.second.begin());
                    std::copy(value.begin(), value.end(), std::get<2>(res.first->second).begin());
                    
                    // Update metadata for the current version
                    update_metadata_on_insert(Index, current_versions[Index], key, height);
                }
                return res.second;
            } catch (boost::interprocess::bad_alloc const& e) {
                log_print("Error inserting into container {}: {}\n", Index, e.what());
                log_print("Next load factor: {:.2f}\n", next_load_factor<Index>());
                log_print("Current size: {}\n", map.size());
                log_print("Current bucket count: {}\n", map.bucket_count());
                log_print("Current load factor: {:.2f}\n", map.load_factor());
                log_print("Current max load factor: {:.2f}\n", map.max_load_factor());
                
                new_version<Index>();

                log_print("Retrying insert into container {} ...\n", Index);
                log_print("Current size: {}\n", map.size());
                log_print("Current bucket count: {}\n", map.bucket_count());
                log_print("Current load factor: {:.2f}\n", map.load_factor());
                log_print("Current max load factor: {:.2f}\n", map.max_load_factor());                
            }    
            --max_retries;
        }
        log_print("Failed to insert after 3 retries\n");
        throw boost::interprocess::bad_alloc();
    }

    template <size_t Index>
        requires (Index < container_sizes.size())
    bool insert_in_index(utxo_key_t const& key, span_bytes value, uint32_t height) {
        size_t size = value.size();
        if (size > container_sizes[Index]) {
            log_print("Error: value size {} exceeds maximum size for container {} ({})\n", 
                      size, Index, container_sizes[Index]);
            throw std::out_of_range("Value size exceeds maximum container size");
        }


        auto& map = container<Index>();
        if (next_load_factor<Index>() >= map.max_load_factor()) {
            new_version<Index>();
        }
        return insert_in_index_safe<Index>(key, value, height);
    }

    template <size_t... Is>
    size_t erase_in_latest_version_impl(utxo_key_t const& key, uint32_t height, std::index_sequence<Is...>) {
        size_t res = 0;
        size_t attempts = 0;

        auto try_erase = [&](auto I) -> bool {
            attempts++;
            auto const it = container<I>().find(key);
            if (it == container<I>().end()) {
                // log_print("-------------------------------------------------\n");
                // log_print("-------------------------------------------------\n");
                // log_print("-------------------------------------------------\n");
                // log_print("-------------------------------------------------\n");
                // log_print("-------------------------------------------------\n");
                // log_print("-------------------------------------------------\n");
                // log_print("Element NOT found in container {} in latest version\n", I());
                // log_print("-------------------------------------------------\n");
                // log_print("-------------------------------------------------\n");
                // log_print("-------------------------------------------------\n");
                // log_print("-------------------------------------------------\n");
                // log_print("-------------------------------------------------\n");
                // log_print("-------------------------------------------------\n");
                return false; // Not found in this version
            }
            auto const insertion_height = std::get<0>(it->second); // Get the height from the value
            // Perform the erase
            container<I>().erase(it);
            // Record the successful erase in current version
            search_stats_.search_records.emplace_back(height, insertion_height, 0, false, true, 'e');
            res = 1; // Update result to indicate successful erase
            return true; // Erase successful
        };

        (try_erase(std::integral_constant<size_t, Is>{}) || ...);

        return res;
    }

    size_t erase_in_latest_version(utxo_key_t const& key, uint32_t height) {
        constexpr auto N = container_sizes.size();
        return erase_in_latest_version_impl(key, height, std::make_index_sequence<N>{});
    }

    template <size_t Index>
        requires (Index < container_sizes.size())    
    size_t erase_x_in_prev_versions(utxo_key_t const& key, uint32_t height) {
        size_t version = current_versions[Index];
        size_t versions_back = 0;
        size_t version_original = version;

        while (version > 0) {
            --version;
            ++versions_back;

            // Check metadata first to avoid unnecessary file access
            if (file_metadata_[Index].size() > version) {
                const auto& metadata = file_metadata_[Index][version];
                if (!metadata.key_in_range(key)) {
                    // Key is definitely not in this file, skip it
                    continue;
                }
            } else {
                // Metadata not available, we have to check the file
                log_print("Warning: metadata for container {} version {} not available, checking file directly\n", Index, version);
                throw std::runtime_error("Metadata not available for version");
            }

            auto const file_name = fmt::format(file_format, db_path_.string(), Index, version);
            auto [map, was_cache_hit] = file_cache_.get_or_open_file<Index>(file_name);

            // Call maintenance periodically
            file_cache_.maintain_cache();

            auto const it = map.find(key);
            if (it == map.end()) {
                continue; // Not found in this version, continue to next
            }
            auto const insertion_height = std::get<0>(it->second);
            map.erase(it);
            
            // Update metadata to reflect the deletion
            update_metadata_on_delete(Index, version);
            
            search_stats_.search_records.emplace_back(height, insertion_height, versions_back, was_cache_hit, true, 'e');
            return 1;
        }

        if (versions_back > 0) {
            search_stats_.search_records.emplace_back(height, 0, versions_back, false, false, 'e');
        }
        return 0;
    }

    size_t get_index_from_size(size_t size) const {
        for (size_t i = 0; i < container_sizes.size(); ++i) {
            if (size <= container_sizes[i]) {
                return i;
            }
        }
        throw std::out_of_range("Size exceeds maximum container size");
    }

    size_t find_latest_version_from_files(size_t index) {
        size_t version = 0;
        std::string file_name = fmt::format(file_format, db_path_.string(), index, version);
        while (std::filesystem::exists(file_name)) {
            ++version;
            file_name = fmt::format(file_format, db_path_.string(), index, version);
        }
        return version;
    }


    template <size_t Index>
        requires (Index < container_sizes.size())
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
                using temp_map_t = map_t<Index>;
                auto* map = segment.find_or_construct<temp_map_t>("temp_map")(
                    mid,
                    key_hash{},
                    key_equal{},
                    segment.get_allocator<std::pair<utxo_key_t const, utxo_val_t<container_sizes[Index]>>>()
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


    template <size_t Index>
        requires (Index < container_sizes.size())
    void open_or_create_container(size_t version) {
        log_print("Opening or creating container {} version {} ...\n", Index, version);

        auto file_name = fmt::format(file_format, db_path_.string(), Index, version);
        log_print("File name: {}\n", file_name);

        try {
            segments_[Index] = std::make_unique<bip::managed_mapped_file>(
                bip::open_or_create, 
                file_name.c_str(), 
                file_sizes[Index]
            );
        } catch (boost::interprocess::bad_alloc const& e) {
            log_print("Error creating memory-mapped file: {}\n", e.what());
            throw;
        }

        auto buckets = min_buckets_ok_[Index];
        while (true) {
            log_print("Creating map with {} buckets ...\n", buckets);
            try {
                auto* map = segments_[Index]->find_or_construct<map_t<Index>>("db_map")(
                    buckets,
                    key_hash{}, 
                    key_equal{},
                    segments_[Index]->get_allocator<std::pair<utxo_key_t const, utxo_val_t<container_sizes[Index]>>>()
                );
        
                containers_[Index] = static_cast<void*>(map);

                log_print("Container {} version {} created with {} buckets\n", Index, version, buckets);

                break;
            } catch (boost::interprocess::bad_alloc const& e) {
                log_print("Error creating map for file {}: {}\n", file_name, e.what());
            }
            buckets /= 2;
        }
        min_buckets_ok_[Index] = buckets;
    }

    template <size_t Index>
        requires (Index < container_sizes.size())
    void close_container() {
        log_print("Closing container {} ...\n", Index);
        if (segments_[Index]) {
            // Save metadata before closing
            save_metadata_to_disk(Index, current_versions[Index]);
            
            segments_[Index]->flush();
            segments_[Index].reset();
            containers_[Index] = nullptr;
        }
    }

    // Batch deletion support
    struct deletion_batch {
        std::vector<std::pair<utxo_key_t, uint32_t>> pending_deletions;
        std::unordered_set<std::string> cached_files;  // Track which files are cached
        
        void add_deletion(const utxo_key_t& key, uint32_t height) {
            pending_deletions.emplace_back(key, height);
        }
        
        void clear() {
            pending_deletions.clear();
            cached_files.clear();
        }
        
        size_t size() const {
            return pending_deletions.size();
        }
        
        bool empty() const {
            return pending_deletions.empty();
        }
    };

    // Enhanced erase function with batching
    size_t erase_batch(const std::vector<std::pair<utxo_key_t, uint32_t>>& keys_to_delete) {
        if (keys_to_delete.empty()) return 0;
        
        size_t total_deleted = 0;
        deletion_batch batch;
        
        // Phase 1: Try to delete from latest version first
        for (const auto& [key, height] : keys_to_delete) {
            auto deleted = erase_in_latest_version(key, height);
            if (deleted > 0) {
                total_deleted += deleted;
            } else {
                // Not found in latest, add to batch for previous versions
                batch.add_deletion(key, height);
            }
        }
        
        // Phase 2: Process remaining deletions in batches, prioritizing cached files
        if (!batch.empty()) {
            total_deleted += erase_batch_from_previous_versions(batch);
        }
        
        return total_deleted;
    }

    // Single key erase (maintains compatibility)
    size_t erase(utxo_key_t const& key, uint32_t height) {
        // Try latest version first
        auto res = erase_in_latest_version(key, height);
        if (res > 0) {
            return res;
        }
        
        // Fall back to single-key previous version search
        constexpr auto N = container_sizes.size();
        return erase_prev_versions_impl(key, height, std::make_index_sequence<N>{});
    }

private:
    // ...existing private members...
    
    // Enhanced method to process deletions from previous versions in batches
    size_t erase_batch_from_previous_versions(deletion_batch& batch) {
        constexpr auto N = container_sizes.size();
        size_t total_deleted = 0;
        
        // Process each container type
        for (size_t container_idx = 0; container_idx < N; ++container_idx) {
            total_deleted += erase_batch_from_container(batch, container_idx);
        }
        
        return total_deleted;
    }
    
    // Process batch deletions for a specific container
    template<size_t Index>
    size_t erase_batch_from_container(deletion_batch& batch, std::integral_constant<size_t, Index>) {
        if (batch.empty()) return 0;
        
        size_t total_deleted = 0;
        std::vector<std::pair<utxo_key_t, uint32_t>> remaining_deletions;
        
        // Get all available versions for this container, starting from most recent
        size_t max_version = current_versions[Index];
        
        // Process versions from newest to oldest
        for (size_t version = max_version; version > 0; --version) {
            size_t actual_version = version - 1;
            
            // Check if we have any keys that could be in this version
            std::vector<std::pair<utxo_key_t, uint32_t>> version_candidates;
            
            for (const auto& [key, height] : batch.pending_deletions) {
                // Use metadata to filter keys that could be in this version
                if (file_metadata_[Index].size() > actual_version) {
                    const auto& metadata = file_metadata_[Index][actual_version];
                    if (metadata.key_in_range(key)) {
                        version_candidates.emplace_back(key, height);
                    }
                }
            }
            
            if (version_candidates.empty()) {
                continue; // No candidates for this version
            }
            
            // Check if file is cached
            auto const file_name = fmt::format(file_format, db_path_.string(), Index, actual_version);
            bool is_cached = batch.cached_files.find(file_name) != batch.cached_files.end();
            
            // Process this version if it's cached or if we have many candidates
            if (is_cached || version_candidates.size() >= 10) {
                auto [map, was_cache_hit] = file_cache_.get_or_open_file<Index>(file_name);
                
                if (was_cache_hit || !is_cached) {
                    batch.cached_files.insert(file_name);
                }
                
                // Process all candidates for this version
                for (auto it = version_candidates.begin(); it != version_candidates.end();) {
                    const auto& [key, height] = *it;
                    
                    auto map_it = map.find(key);
                    if (map_it != map.end()) {
                        auto const insertion_height = std::get<0>(map_it->second);
                        map.erase(map_it);
                        
                        // Update metadata
                        update_metadata_on_delete(Index, actual_version);
                        
                        // Record successful deletion
                        size_t versions_back = max_version - actual_version;
                        search_stats_.search_records.emplace_back(height, insertion_height, versions_back, was_cache_hit, true, 'e');
                        
                        total_deleted++;
                        it = version_candidates.erase(it);
                    } else {
                        ++it;
                    }
                }
                
                // Remove processed keys from batch
                auto& pending = batch.pending_deletions;
                pending.erase(
                    std::remove_if(pending.begin(), pending.end(),
                        [&version_candidates](const auto& pending_item) {
                            return std::find(version_candidates.begin(), version_candidates.end(), pending_item) == version_candidates.end();
                        }),
                    pending.end()
                );
            }
        }
        
        return total_deleted;
    }
    
    size_t erase_batch_from_container(deletion_batch& batch, size_t container_idx) {
        switch (container_idx) {
            case 0: return erase_batch_from_container(batch, std::integral_constant<size_t, 0>{});
            case 1: return erase_batch_from_container(batch, std::integral_constant<size_t, 1>{});
            case 2: return erase_batch_from_container(batch, std::integral_constant<size_t, 2>{});
            case 3: return erase_batch_from_container(batch, std::integral_constant<size_t, 3>{});
            default: return 0;
        }
    }
    
    // Enhanced new_version that moves latest file to cache
    template <size_t Index>
        requires (Index < container_sizes.size())
    void new_version() {
        // Before closing, move the current file to cache to optimize for upcoming deletions
        if (segments_[Index]) {
            auto current_file_name = fmt::format(file_format, db_path_.string(), Index, current_versions[Index]);
            
            // Check if we can add it to cache
            auto [map, was_already_cached] = file_cache_.try_add_to_cache<Index>(current_file_name, std::move(segments_[Index]), containers_[Index]);
            
            if (!was_already_cached) {
                log_print("new_version: moved latest file {} to cache\n", current_file_name);
            }
        }
        
        // Save metadata before closing
        save_metadata_to_disk(Index, current_versions[Index]);
        
        // Clear current references (file is now managed by cache)
        containers_[Index] = nullptr;
        
        // Increment version and open new container
        ++current_versions[Index];
        open_or_create_container<Index>(current_versions[Index]);
        
        // Initialize metadata for the new version
        load_metadata_from_disk(Index, current_versions[Index]);
    }

    // ...existing private methods...
};


// utxo_db db_;
// // std::vector<utxo_key_t> all_utxos_;
// using utxo_vector_t = std::vector<std::pair<utxo_key_t, uint32_t>>;
// utxo_vector_t all_utxos_;

// void print_hex(uint8_t const* data, size_t size) {
//     for (size_t i = 0; i < size; ++i) {
//         std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(data[i]);
//     }
//     std::cout << std::dec << std::endl;
// }

// std::vector<utxo_entry_static> generate_utxos_static(size_t count) {
//     std::vector<utxo_entry_static> utxos;
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<uint8_t> byte_dist(0, 255);
//     std::uniform_int_distribution<size_t> val_size_dist(UTXO_MIN_VALUE, UTXO_MAX_VALUE);

//     for (size_t i = 0; i < count; ++i) {
//         utxo_key_t key;
//         for (size_t j = 0; j < UTXO_KEY_SIZE; ++j) {
//             key[j] = byte_dist(gen);
//         }

//         size_t const val_size = val_size_dist(gen);
//         utxo_value_static_t value{val_size, {}};
//         for (size_t j = 0; j < val_size; ++j) {
//             value.second[j] = byte_dist(gen);
//         }
//         value.second[10] = 0xFF;
//         value.second[11] = 0xFE;
//         value.second[12] = 0xFD;
//         value.second[13] = 0xFC;
//         // print_hex(value.second.data(), UTXO_MAX_VALUE);

//         utxos.emplace_back(std::move(key), std::move(value));
//     }
//     return utxos;
// }

// void open_db() {

//     log_print("Opening DB ...\n");

//     db_.configure(
//         "utxo_interprocess_multiple",
//         true
//     ); 
    
//     log_print("DB opened ...\n");
//     // log_print("DB opened with size: {}\n", db_map->size());
// }

// void open_db_existing() {
//     log_print("Opening Existing DB ...\n");

//     db_.configure(
//         "utxo_interprocess_multiple",
//         false
//     ); 

//     log_print("DB opened ...\n");
//     // log_print("DB opened with size: {}\n", db_map->size());
// }

// void close_db() {
//     log_print("Closing DB... \n");
//     db_.close();
// }

// bool check_db_integrity() {
//     // size_t i = 0;
//     // utxo_key_t prev_k;
//     // for (auto const& [k, v] : *db_map) {
//     //     prev_k = k;
//     //     if (v.second[10] != 0xFF) {
//     //         break;
//     //     }
//     //     if (v.second[11] != 0xFE) {
//     //         break;
//     //     }
//     //     if (v.second[12] != 0xFD) {
//     //         break;
//     //     }
//     //     if (v.second[13] != 0xFC) {
//     //         break;
//     //     }
//     //     ++i;
//     // }
//     // if (i == db_map->size()) {
//     //     log_print("DB integrity check passed: {} entries are valid.\n", i);
//     //     return true;
//     // } else {
//     //     log_print("DB integrity check failed: entry {} is invalid.\n", i);
//     //     auto kv = *db_map->find(prev_k);
//     //     print_hex(kv.second.second.data(), UTXO_MAX_VALUE);
//     //     return false;
//     // }

//     return true;
// }


// utxo_vector_t get_to_delete_entries_real_world(size_t num_to_delete) {
//     log_print("Found {} entries in the database\n", all_utxos_.size());

//     // Ordenamos los UTXOs por su "antigüedad", asumimos que el valor (second) representa antigüedad
//     std::sort(all_utxos_.begin(), all_utxos_.end(), [](auto const& x, auto const& y){
//         return x.second < y.second;
//     });

//     utxo_vector_t res;
//     res.reserve(num_to_delete);

//     // Porcentajes que representan las proporciones de eliminaciones
//     std::array<float, 4> percents = {0.7, 0.2, 0.07, 0.03};

//     size_t total_utxos = all_utxos_.size();
//     size_t num_selected = 0;
//     std::vector<std::pair<size_t, size_t>> ranges_to_remove;

//     for (size_t i = 0; i < percents.size(); ++i) {
//         // Calculamos el rango correspondiente
//         size_t range_start = (total_utxos * i) / percents.size();
//         size_t range_end = (total_utxos * (i + 1)) / percents.size();
//         size_t range_size = range_end - range_start;
//         // log_print("Range {}: {} to {}: size {}\n", i, range_start, range_end, range_size);

//         // Calculamos cuántos elementos se seleccionan de este rango
//         size_t num_from_range = std::min(size_t(num_to_delete * percents[i]), range_size);
//         // log_print("num_to_delete * percents[i]: {} * {} = {}\n", num_to_delete, percents[i], size_t(num_to_delete * percents[i]));
//         // log_print("num_from_range: {}\n", num_from_range);
//         num_selected += num_from_range;

//         // // Seleccionamos aleatoriamente de este rango
//         // std::vector<utxo_key_t> temp_range;
//         // temp_range.reserve(range_size);
//         // for (size_t j = range_start; j < range_end; ++j) {
//         //     temp_range.push_back(all_utxos_[j].first);
//         // }

//         // std::shuffle(temp_range.begin(), temp_range.end(), std::mt19937{std::random_device{}()});
//         // temp_range.resize(num_from_range);

//         // // Agregamos al resultado
//         // res.insert(res.end(), temp_range.begin(), temp_range.end());

//         std::shuffle(
//             all_utxos_.begin() + range_start, 
//             all_utxos_.begin() + range_end, 
//             std::mt19937{std::random_device{}()}
//         );

//         std::move(
//             all_utxos_.begin() + range_start, 
//             all_utxos_.begin() + range_start + num_from_range, 
//             std::back_inserter(res)
//         );
//         ranges_to_remove.emplace_back(range_start, range_start + num_from_range);

//         // Si ya hemos seleccionado el total requerido, salimos
//         if (res.size() >= num_to_delete) {
//             break;
//         }
//     }
//     // log_print("Ranges to remove before sorting:\n");
//     // for (const auto& [start, end] : ranges_to_remove) {
//     //     log_print("  {} to {}\n", start, end);
//     // }

//     std::sort(ranges_to_remove.begin(), ranges_to_remove.end(), 
//               [](auto const& a, auto const& b) { return a.first > b.first; });

//     // log_print("Ranges to remove after sorting:\n");
//     // for (const auto& [start, end] : ranges_to_remove) {
//     //     log_print("  {} to {}\n", start, end);
//     // }

//     // log_print("all_utxos_.size() before removal: {}\n", all_utxos_.size());
//     // Eliminamos los rangos seleccionados de all_utxos_    
//     for (const auto& [start, end] : ranges_to_remove) {
//         all_utxos_.erase(all_utxos_.begin() + start, all_utxos_.begin() + end);
//     }
//     // log_print("all_utxos_.size() after removal: {}\n", all_utxos_.size());


//     // // Si aún falta para llegar a num_to_delete (por redondeos), completamos
//     // if (res.size() < num_to_delete) {
//     //     std::vector<utxo_key_t> remaining;
//     //     for (auto const& [key, _] : all_utxos_) {
//     //         if (std::find(res.begin(), res.end(), key) == res.end()) {
//     //             remaining.push_back(key);
//     //         }
//     //     }

//     //     std::shuffle(remaining.begin(), remaining.end(), std::mt19937{std::random_device{}()});
//     //     size_t remaining_needed = num_to_delete - res.size();
//     //     res.insert(res.end(), remaining.begin(), remaining.begin() + remaining_needed);
//     // }

//     return res;
// }


// BenchResult do_insertion(size_t num_entries, uint32_t height) {
//     log_print("Generating {} UTXOs...\n", num_entries);
//     // auto utxos = generate_utxos_static(num_entries);
//     auto utxos = generate_utxos(num_entries);

//     auto insert_start = std::chrono::high_resolution_clock::now();

//     for (const auto& [k, v] : utxos) {
//         db_.insert(k, v, height);
//     }

//     auto insert_end = std::chrono::high_resolution_clock::now();
//     auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(insert_end - insert_start).count();
//     auto duration_us = duration_ns / 1000.0;
//     auto duration_s = duration_ns / 1000000000.0;
    
//     for (const auto& [k, v] : utxos) {
//         all_utxos_.emplace_back(k, height);
//     }

//     log_print("[Interprocess] Inserted {} entries in {:.2f} µs ({:.6f} s)\n", num_entries, duration_us, duration_s);
    
//     return {duration_ns, num_entries};
// }

// BenchResult do_deletion(size_t num_to_delete, uint32_t height) {
//     log_print("Deleting {} entries ...\n", num_to_delete);
//     // auto const to_delete = get_to_delete_entries(num_to_delete);
//     auto const to_delete = get_to_delete_entries_real_world(num_to_delete);
//     log_print("Deleting {} entries ...\n", to_delete.size());
//     // log_print("Deleting {} entries ...\n", 0);
    
//     auto delete_start = std::chrono::high_resolution_clock::now();

//     for (const auto& [k, _h] : to_delete) {
//         db_.erase(k, height);
//     }

//     auto delete_end = std::chrono::high_resolution_clock::now();

//     auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(delete_end - delete_start).count();
//     auto duration_us = duration_ns / 1000.0;
//     auto duration_s = duration_ns / 1000000000.0;
    
//     log_print("[Interprocess] Deleted {} entries in {:.2f} µs ({:.6f} s)\n", 
//                to_delete.size(), duration_us, duration_s);
    
//     return {duration_ns, to_delete.size()};


//     // log_print("[Interprocess] Deleted {} entries in {:.2f} µs ({:.6f} s)\n", 
//     //            0, duration_us, duration_s);
    
//     // return {duration_ns, 0};

// }

// MapStats get_db_stats() {
//     MapStats stats;

//     // stats["num_entries"] = db_map->size();
//     // stats["bucket_count"] = db_map->bucket_count();
//     // stats["load_factor"] = db_map->load_factor();
//     // stats["max_load_factor"] = db_map->max_load_factor();

//     return stats;
// }

// void print_interprocess_stats() {
//     log_print("\n=== INTERPROCESS SEARCH STATISTICS ===\n");
    
//     const auto& search_stats = db_.get_search_stats();
//     auto all_stats = search_stats.get_all_statistics();
    
//     if (all_stats.total_operations == 0) {
//         log_print("No search operations recorded.\n");
//         return;
//     }
    
//     log_print("Total Operations: {}\n", all_stats.total_operations);
//     log_print("  Find operations: {} ({:.1f}%)\n", 
//               all_stats.find_operations, 
//               100.0 * all_stats.find_operations / all_stats.total_operations);
//     log_print("  Erase operations: {} ({:.1f}%)\n", 
//               all_stats.erase_operations, 
//               100.0 * all_stats.erase_operations / all_stats.total_operations);
    
//     log_print("\nSearch Results:\n");
//     log_print("  Found: {} ({:.1f}%)\n", 
//               all_stats.found_operations, 
//               100.0 * all_stats.found_operations / all_stats.total_operations);
//     log_print("  Not Found: {} ({:.1f}%)\n", 
//               all_stats.not_found_operations, 
//               100.0 * (1.0 - all_stats.found_operations / all_stats.total_operations));
//     log_print("  Cache hits: {} ({:.1f}%)\n", 
//               all_stats.cache_hits, 
//               100.0 * all_stats.cache_hits / all_stats.total_operations);
    
//     log_print("\nVersion Access Patterns:\n");
//     log_print("  Current version hits: {} ({:.1f}%)\n", 
//               all_stats.current_version_hits, 
//               100.0 * all_stats.current_version_hits / all_stats.total_operations);
//     log_print("  Previous version accesses: {} ({:.1f}%)\n", 
//               all_stats.prev_version_accesses, 
//               100.0 * all_stats.prev_version_accesses / all_stats.total_operations);
//     log_print("  Average depth: {:.2f}\n", all_stats.avg_depth);
//     log_print("  Maximum depth: {}\n", all_stats.max_depth);
    
//     if (all_stats.cache_accesses > 0) {
//         log_print("\nCache Performance:\n");
//         log_print("  Cache accesses: {}\n", all_stats.cache_accesses);
//         log_print("  Cache hits: {} ({:.1f}%)\n", 
//                   all_stats.cache_hits, 
//                   100.0 * all_stats.cache_hit_rate);
//     }
    
//     if (all_stats.found_operations > 0) {
//         log_print("\nUTXO Age Analysis:\n");
//         log_print("  Average UTXO age: {:.2f} blocks\n", all_stats.avg_utxo_age);
//         log_print("  Min UTXO age: {} blocks\n", all_stats.min_utxo_age);
//         log_print("  Max UTXO age: {} blocks\n", all_stats.max_utxo_age);
//     }
    
//     log_print("\n");
// }
