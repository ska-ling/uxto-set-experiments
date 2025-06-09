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
using segment_manager_t = bip::managed_mapped_file::segment_manager;
using key_hash = boost::hash<utxo_key_t>;
using key_equal = std::equal_to<utxo_key_t>;

// Select appropriate uint type for size
template<size_t Size>
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
template<size_t Size>
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
    size_t attempt_count = 0;
    boost::unordered_flat_set<std::string> tried_files;
    std::chrono::steady_clock::time_point last_attempt;
    
    static constexpr size_t max_attempts = 3;
    static constexpr auto retry_interval = std::chrono::seconds{5};
    
    deferred_deletion_entry(utxo_key_t const& k, uint32_t h) 
        : key(k), height(h), last_attempt(std::chrono::steady_clock::now()) {}
    
    bool should_retry() const { 
        return attempt_count < max_attempts; 
    }
    
    bool can_retry_now() const {
        return std::chrono::steady_clock::now() - last_attempt >= retry_interval;
    }
    
    void mark_tried(std::string const& file_path) {
        tried_files.insert(file_path);
        attempt_count++;
        last_attempt = std::chrono::steady_clock::now();
    }
};

// File cache - keeping your implementation but cleaner
class file_cache {
public:
    explicit file_cache(size_t max_size = 1) 
        : max_cached_files_(max_size) 
    {}
    
    template<size_t Index>
    std::pair<utxo_map<container_sizes[Index]>&, bool> get_or_open_file(std::string const& file_path) {
        ++gets_;
        auto const now = std::chrono::steady_clock::now();
        
        // Track access
        ++access_frequency_[file_path];
        
        // Check cache
        if (auto it = cache_.find(file_path); it != cache_.end()) {
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
            auto segment = std::make_unique<bip::managed_mapped_file>(
                bip::open_only, file_path.c_str());
            
            auto* map = segment->find<utxo_map<container_sizes[Index]>>("db_map").first;
            if (!map) {
                throw std::runtime_error("Map not found in file");
            }
            
            cache_[file_path] = {
                .segment = std::move(segment),
                .map_ptr = map,
                .last_used = now,
                .access_count = 1,
                .is_pinned = false
            };
            
            return {*map, false};
        } catch (std::exception const& e) {
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
    
private:
    struct cached_file {
        std::unique_ptr<bip::managed_mapped_file> segment;
        void* map_ptr;
        std::chrono::steady_clock::time_point last_used;
        size_t access_count = 0;
        bool is_pinned = false;
    };
    
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
    
    boost::unordered_flat_map<std::string, cached_file> cache_;
    boost::unordered_flat_map<std::string, size_t> access_frequency_;
    size_t max_cached_files_;
    size_t gets_ = 0;
    size_t hits_ = 0;
    size_t evictions_ = 0;
};

// Main database class
class utxo_db {
    using span_bytes = std::span<uint8_t const>;

    template <size_t Index>
        requires (Index < container_sizes.size())
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
        
        auto optimal0 = find_optimal_buckets<0>("./optimal", file_sizes[0], 7864304);
        log_print("Optimal number of buckets for container {} and file size {}: {}\n", 0, file_sizes[0], optimal0);
        auto optimal1 = find_optimal_buckets<1>("./optimal", file_sizes[1], 7864304);
        log_print("Optimal number of buckets for container {} and file size {}: {}\n", 1, file_sizes[1], optimal1);
        auto optimal2 = find_optimal_buckets<2>("./optimal", file_sizes[2], 7864304);
        log_print("Optimal number of buckets for container {} and file size {}: {}\n", 2, file_sizes[2], optimal2);
        auto optimal3 = find_optimal_buckets<3>("./optimal", file_sizes[3], 7864304);
        log_print("Optimal number of buckets for container {} and file size {}: {}\n", 3, file_sizes[3], optimal3);

        // Initialize containers
        for_each_index<container_sizes.size()>([&](auto I) {
            size_t latest_version = find_latest_version_from_files(I);
            open_or_create_container<I>(latest_version);
            
            // Load metadata
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
    
    size_t size() const {
        return entries_count_;
    }

    // Clean insert interface
    bool insert(utxo_key_t const& key, span_bytes value, uint32_t height) {
        size_t const index = get_index_from_size(value.size());
        if (index >= container_sizes.size()) {
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
    
    // Process deferred deletions
    size_t process_pending_deletions(size_t max_to_process = 100) {
        if (deferred_deletions_.empty()) return 0;
        
        size_t processed = 0;
        size_t successful = 0;
        
        for (auto it = deferred_deletions_.begin(); 
             it != deferred_deletions_.end() && processed < max_to_process;) {
            
            if (!it->can_retry_now()) {
                ++it;
                continue;
            }
            
            if (!it->should_retry()) {
                it = deferred_deletions_.erase(it);
                processed++;
                continue;
            }
            
            // Try to delete
            bool deleted = false;
            for_each_index<container_sizes.size()>([&](auto I) {
                if (!deleted) {
                    deleted = try_delete_deferred<I>(*it);
                }
            });
            
            if (deleted) {
                successful++;
                it = deferred_deletions_.erase(it);
            } else {
                it->mark_tried("");
                ++it;
            }
            processed++;
        }
        
        return successful;
    }
    
private:
    static constexpr std::string_view file_format = "{}/cont_{}_v{:05}.dat";
    
    // Storage
    fs::path db_path_ = "utxo_interprocess";
    std::array<std::unique_ptr<bip::managed_mapped_file>, container_sizes.size()> segments_;
    std::array<void*, container_sizes.size()> containers_{};
    std::array<size_t, container_sizes.size()> current_versions_ = {};
    std::array<size_t, container_sizes.size()> min_buckets_ok_ = {
        3932159,
        1966079,
        245759,
        7679
    };
    size_t entries_count_ = 0; // Total entries across all containers
    
    // Metadata and caching
    std::array<std::vector<file_metadata>, container_sizes.size()> file_metadata_;
    file_cache file_cache_; // number of cached files. TODO: change
    search_stats search_stats_;
    std::vector<deferred_deletion_entry> deferred_deletions_;
    
    // Get container
    template<size_t Index>
    utxo_map<container_sizes[Index]>& container() {
        return *static_cast<utxo_map<container_sizes[Index]>*>(containers_[Index]);
    }
    
    // Insert implementation
    template <size_t Index>
    bool insert_in_index(utxo_key_t const& key, span_bytes value, uint32_t height) {
        // Check if rotation needed
        if (should_rotate<Index>()) {
            log_print("Rotating container {} due to load factor\n", Index);
            new_version<Index>();
        }
        auto& map = container<Index>();
        
        // Insert
        utxo_value<container_sizes[Index]> val;
        val.block_height = height;
        val.set_data(value);
        
        size_t max_retries = 3;
        while (max_retries > 0) {
            try {
                auto [it, inserted] = map.emplace(key, val);
                if (inserted) {
                    ++entries_count_;
                    update_metadata_on_insert(Index, current_versions_[Index], key, height);
                }
                return inserted;
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
    bool should_rotate() const {
        auto const& map = container<Index>();
        if (map.bucket_count() == 0) return false;
        float next_load = float(map.size() + 1) / float(map.bucket_count());
        return next_load >= map.max_load_factor();
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

    // Find in latest version
    std::optional<std::vector<uint8_t>> find_in_latest_version(utxo_key_t const& key, uint32_t height) {
        std::optional<std::vector<uint8_t>> result;
        
        for_each_index<container_sizes.size()>([&](auto I) {
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
        
        for_each_index<container_sizes.size()>([&](auto I) {
            if (!result) {
                result = find_in_prev_versions<I>(key, height);
            }
        });
        
        if (!result) {
            search_stats_.add_record(height, 0, 1, false, false, 'f');
        }
        
        return result;
    }
    
    template<size_t Index>
    std::optional<std::vector<uint8_t>> find_in_prev_versions(utxo_key_t const& key, uint32_t height) {
        for (size_t v = current_versions_[Index]; v-- > 0;) {
            // Check metadata
            if (file_metadata_[Index].size() > v && !file_metadata_[Index][v].key_in_range(key)) {
                continue;
            }
            
            auto file_name = fmt::format(file_format, db_path_.string(), Index, v);
            auto [map, cache_hit] = file_cache_.get_or_open_file<Index>(file_name);
            
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
        
        for_each_index<container_sizes.size()>([&](auto I) {
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
        auto it = std::ranges::find_if(deferred_deletions_,
            [&key](auto const& entry) { return entry.key == key; });
        
        if (it == deferred_deletions_.end()) {
            deferred_deletions_.emplace_back(key, height);
            log_print("deferred_deletion: added UTXO for later processing (total: {})\n", 
                     deferred_deletions_.size());
        }
    }
    
    template<size_t Index>
    bool try_delete_deferred(deferred_deletion_entry& entry) {
        for (size_t v = current_versions_[Index]; v-- > 0;) {
            auto file_name = fmt::format(file_format, db_path_.string(), Index, v);
            
            if (entry.tried_files.contains(file_name)) continue;
            
            // Check metadata
            if (file_metadata_[Index].size() > v && !file_metadata_[Index][v].key_in_range(entry.key)) {
                entry.tried_files.insert(file_name);
                continue;
            }
            
            try {
                auto [map, cache_hit] = file_cache_.get_or_open_file<Index>(file_name);
                if (map.erase(entry.key) > 0) {
                    update_metadata_on_delete(Index, v);
                    size_t depth = current_versions_[Index] - v;
                    search_stats_.add_record(entry.height, 0, depth, cache_hit, true, 'e');
                    return true;
                }
            } catch (...) {
                // Continue to next file
            }
            
            entry.tried_files.insert(file_name);
        }
        
        return false;
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

    // File management
    template<size_t Index>
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
    
    template<size_t Index>
    void close_container() {
        if (segments_[Index]) {
            save_metadata_to_disk(Index, current_versions_[Index]);
            segments_[Index]->flush();
            segments_[Index].reset();
            containers_[Index] = nullptr;
        }
    }
    
    template<size_t Index>
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
        for (size_t i = 0; i < container_sizes.size(); ++i) {
            if (size <= container_sizes[i]) return i;
        }
        return container_sizes.size();
    }
    
    size_t find_latest_version_from_files(size_t index) {
        size_t version = 0;
        while (fs::exists(fmt::format(file_format, db_path_.string(), index, version))) {
            ++version;
        }
        return version > 0 ? version - 1 : 0;
    }
};

} // namespace utxo