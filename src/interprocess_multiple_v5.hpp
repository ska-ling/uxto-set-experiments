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
    400_mib,
    200_mib,
    100_mib
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
        fmt::print("{:02x}", key[31 - i]);
    }   
    // the last 4 bytes are the output index, print as integer
    uint32_t output_index = 0;
    std::copy(key.end() - 4, key.end(), reinterpret_cast<uint8_t*>(&output_index));
    fmt::print(":{}", output_index);
    fmt::print("\n");
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
        return fmt::format("{}/cont_{}_v{:05}.dat", base_path_, index, version);
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
      
public:
    void configure(std::string_view path, bool remove_existing = false) {
        db_path_ = path;
        
        if (remove_existing && fs::exists(path)) {
            fs::remove_all(path);
        }
        fs::create_directories(path);
        
        // Configure file cache with base path
        file_cache_ = file_cache(std::string(path));
        
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
            return insert_in_index<decltype(ic)::value>(key, value, height);
        }, make_index_variant(index));
    }
    
    // Clean erase interface with deferred deletion
    size_t erase(utxo_key_t const& key, uint32_t height) {
        // Try current version first
        if (auto res = erase_in_latest_version(key, height); res > 0) {
            entries_count_ -= res;
            return res;
        }
        
        // Try cached files only
        if (auto res = erase_from_cached_files_only(key, height); res > 0) {
            entries_count_ -= res;
            return res;
        }
        
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
    // Returns list of UTXOs that could not be deleted (errors that must be reported)
    std::vector<utxo_key_t> process_all_pending_deletions() {
        if (deferred_deletions_.empty()) return {};
        
        size_t initial_size = deferred_deletions_.size();
        log_print("Processing ALL {} deferred deletions - MUST complete entirely\n", initial_size);
        
        size_t successful_deletions = 0;
        
        // Phase 1: Process ALL cached files first to maximize cache efficiency
        auto cached_files = file_cache_.get_cached_files();
        if (!cached_files.empty()) {
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
                    
                    auto file_name = fmt::format(file_format, db_path_.string(), I.value, v);
                    if (!fs::exists(file_name)) {
                        continue;
                    }
                    
                    successful_deletions += process_deferred_deletions_in_file(I.value, v, false);
                }
            });
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
        
        if (!failed_deletions.empty()) {
            log_print("ERROR: {} UTXOs could not be deleted - these are processing errors!\n", 
                     failed_deletions.size());
        }
        
        return failed_deletions;
    }

    // Simplified deferred deletion processing (kept for compatibility if needed)
    size_t process_pending_deletions(size_t max_to_process = SIZE_MAX) {
        auto failed_deletions = process_all_pending_deletions();
        // For compatibility, we don't return the failed list here
        // but the new API should use process_all_pending_deletions() directly
        return deferred_deletions_.size() > 0 ? 0 : max_to_process;
    }
    
private:
    static constexpr std::string_view file_format = "{}/cont_{}_v{:05}.dat";
    
    // Storage
    fs::path db_path_ = "utxo_interprocess";
    std::array<std::unique_ptr<bip::managed_mapped_file>, IdxN> segments_;
    std::array<void*, IdxN> containers_{};
    std::array<size_t, IdxN> current_versions_ = {};
    // std::array<size_t, IdxN> min_buckets_ok_ = {
    //     3932159,
    //     1966079,
    //      245759,
    //        7679
    // };

    std::array<size_t, IdxN> min_buckets_ok_ = {};
    size_t entries_count_ = 0; // Total entries across all containers
    
    // Metadata and caching
    std::array<std::vector<file_metadata>, IdxN> file_metadata_;
    file_cache file_cache_ = file_cache(""); // number of cached files. TODO: change
    search_stats search_stats_;
    // std::vector<deferred_deletion_entry> deferred_deletions_;
    boost::unordered_flat_set<deferred_deletion_entry> deferred_deletions_;
    
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
                auto [it, inserted] = map.emplace(key, val);
                if (inserted) {
                    ++entries_count_;
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
                    search_stats_.add_record(height, it->second.block_height, 0, false, true, 'e');
                    map.erase(it);
                    result = 1;
                }
            }
        });
        
        return result;
    }
    
    // Erase from cached files only
    size_t erase_from_cached_files_only(utxo_key_t const& key, uint32_t height) {
        // Implementation similar to find but only checking cached files
        // ... (similar pattern as find_in_prev_versions but with cache check)
        return 0;
    }
    
    // Deferred deletion helpers
    void add_to_deferred_deletions(utxo_key_t const& key, uint32_t height) {
        // now deferred_deletions_ is an unordered set
        auto [it, inserted] = deferred_deletions_.emplace(key, height);
        if (inserted) {
            // log_print("deferred_deletion: added UTXO for later processing (total: {})\n",  deferred_deletions_.size());
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
                        search_stats_.add_record(it->height, 0, depth, cache_hit, true, 'e');
                        
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

    // File management
    template <size_t Index>
    void open_or_create_container(size_t version) {
        auto file_name = fmt::format(file_format, db_path_.string(), Index, version);
        
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
        current_versions_[Index]++;
        open_or_create_container<Index>(current_versions_[Index]);
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
        while (fs::exists(fmt::format(file_format, db_path_.string(), index, version))) {
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
};

} // namespace utxo