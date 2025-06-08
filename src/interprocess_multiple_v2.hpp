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

class file_cache {
    struct cached_file {
        std::unique_ptr<bip::managed_mapped_file> segment;
        void* map_ptr;
        std::chrono::steady_clock::time_point last_used;
    };
    
    boost::unordered_flat_map<std::string, cached_file> cache_;
    // size_t max_cached_files_ = 10; // Configurable
    size_t max_cached_files_ = 20; // Configurable

    size_t gets_ = 0;
    size_t hits_ = 0;
    
public:
    float get_hit_rate() const {
        return float(hits_) / float(gets_);
    }

    template <size_t Index>
        requires (Index < container_sizes.size())
    std::pair<map_t<Index>&, bool> get_or_open_file(std::string const& file_path) {
        ++gets_;
        auto const now = std::chrono::steady_clock::now();
        
        // Check if already in cache
        auto const it = cache_.find(file_path);
        if (it != cache_.end()) {
            it->second.last_used = now;
            // log_print("file_cache: file cache hit Ok for {}\n", file_path);
            ++hits_;
            return {*static_cast<map_t<Index>*>(it->second.map_ptr), true};
        }
        
        // Evict least recently used if cache is full
        if (cache_.size() >= max_cached_files_) {
            auto lru = std::min_element(cache_.begin(), cache_.end(),
                [](const auto& a, const auto& b) {
                    return a.second.last_used < b.second.last_used;
                });
            log_print("file_cache: erasing element from the cache: {}\n", lru->first);
            cache_.erase(lru);
        }
        
        // Open new file
        log_print("file_cache: opening a non-cached file for {}\n", file_path);
        auto segment = std::make_unique<bip::managed_mapped_file>(
            bip::open_only, file_path.c_str());
            
        auto* map = segment->find<map_t<Index>>("db_map").first;
        
        // Store in cache
        cache_[file_path] = {std::move(segment), map, now};
        return {*map, false};
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
            open_or_create_container<I>(find_latest_version_from_files(I));
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

        // log_print("******************************************************\n");
        // log_print("******************************************************\n");
        // log_print("******************************************************\n");
        // log_print("******************************************************\n");
        // log_print("******************************************************\n");
        // log_print("res: {} - size_prev: {} - size_current: {}\n", res, size_prev, search_stats_.search_records.size());
        // log_print("******************************************************\n");
        // log_print("******************************************************\n");
        // log_print("******************************************************\n");
        // log_print("******************************************************\n");
        // log_print("******************************************************\n");


        return res;
    }

    size_t erase(utxo_key_t const& key, uint32_t height) {
        constexpr auto N = container_sizes.size();
        auto res = erase_in_latest_version(key, height);
        if (res > 0) {
            return res;
        } else {
            // log_print("********************************************\n");
            // log_print("********************************************\n");
            // log_print("********************************************\n");
            // log_print("********************************************\n");
            // log_print("********************************************\n");
            // log_print("********************************************\n");
            // log_print("********************************************\n");
            // log_print("erase: not found in latest version, trying previous versions\n");
            // log_print("********************************************\n");
            // log_print("********************************************\n");
            // log_print("********************************************\n");
            // log_print("********************************************\n");
            // log_print("********************************************\n");
            // log_print("********************************************\n");
            // log_print("********************************************\n");
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

            auto const file_name = fmt::format(file_format, db_path_.string(), Index, version);
            auto [map, was_cache_hit] = file_cache_.get_or_open_file<Index>(file_name);

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
    bool insert_in_index(utxo_key_t const& key, span_bytes value, uint32_t height) {
        auto& map = container<Index>();
        if (next_load_factor<Index>() >= map.max_load_factor()) {
            new_version<Index>();
        }
        size_t size = value.size();
        if (size > container_sizes[Index]) {
            log_print("Error: value size {} exceeds maximum size for container {} ({})\n", 
                      size, Index, container_sizes[Index]);
            throw std::out_of_range("Value size exceeds maximum container size");
        }

        size_t max_retries = 3;
        while (max_retries > 0) {
            try {
                log_print("Before emplace in DB for Index {}: \n", Index);
                auto res = map.emplace(key, utxo_val_t<container_sizes[Index]>{
                    height,
                    value.size(), 
                    {}
                });
                log_print("After emplace in DB for Index {}: \n", Index);
                if (res.second) {
                    // insert took place
                    // std::copy(value.begin(), value.end(), res.first->second.second.begin());
                    std::copy(value.begin(), value.end(), std::get<2>(res.first->second).begin());
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

        // // If not found in current version, record the failed attempt
        // if (res == 0) {
        //     log_print("******************************************************\n");
        //     log_print("******************************************************\n");
        //     log_print("******************************************************\n");
        //     log_print("******************************************************\n");
        //     log_print("******************************************************\n");
        //     log_print("******************************************************\n");
        //     log_print("******************************************************\n");
        //     log_print("******************************************************\n");
        //     log_print("******************************************************\n");
        //     log_print("******************************************************\n");
        //     log_print("******************************************************\n");
        //     log_print("******************************************************\n");
        //     log_print("******************************************************\n");
        //     log_print("******************************************************\n");
        //     log_print("******************************************************\n");
        //     log_print("******************************************************\n");
        //     log_print("******************************************************\n");
        //     log_print("******************************************************\n");
        //     log_print("******************************************************\n");
        //     search_stats_.search_records.emplace_back(height, 0, 0, false, false, 'e');
        // }

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
        // log_print("trying to erase element in index {}\n", Index);

        while (version > 0) {
            --version;
            ++versions_back;

            auto const file_name = fmt::format(file_format, db_path_.string(), Index, version);
            // log_print("trying to erase element in version {} of index {}\n", version, Index);

            auto [map, was_cache_hit] = file_cache_.get_or_open_file<Index>(file_name);

            auto const it = map.find(key);
            if (it == map.end()) {
                // log_print("element NOT found in version {} of index {}\n", version, Index);
                continue; // Not found in this version, continue to next
            }

            auto const insertion_height = std::get<0>(it->second);
            map.erase(it);
            // log_print("element found in version {} of index {}\n", version, Index);
            search_stats_.search_records.emplace_back(height, insertion_height, versions_back, was_cache_hit, true, 'e');
            return 1;
        }
        // log_print("element NOT found in any version of index {}\n", Index);

        if (versions_back > 0) {
            search_stats_.search_records.emplace_back(height, 0, versions_back, false, false, 'e');
        } else {

            // log_print("******************************************************\n");
            // log_print("******************************************************\n");
            // log_print("******************************************************\n");
            // log_print("******************************************************\n");
            // log_print("******************************************************\n");
            // log_print("versions_back: {} - version_original: {} - Index: {}", versions_back, version_original, Index);
            // log_print("******************************************************\n");
            // log_print("******************************************************\n");
            // log_print("******************************************************\n");
            // log_print("******************************************************\n");
            // log_print("******************************************************\n");

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
            segments_[Index]->flush();
            segments_[Index].reset();
            containers_[Index] = nullptr;
        }
    }

    template <size_t Index>
        requires (Index < container_sizes.size())
    void new_version() {
        // Avoid rehashing, open a new container
        close_container<Index>();
        ++current_versions[Index];
        open_or_create_container<Index>(current_versions[Index]);
    }
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
//               100.0 * all_stats.found_rate);
//     log_print("  Not Found: {} ({:.1f}%)\n", 
//               all_stats.not_found_operations, 
//               100.0 * (1.0 - all_stats.found_rate));
    
//     log_print("\nVersion Access Patterns:\n");
//     log_print("  Current version hits: {} ({:.1f}%)\n", 
//               all_stats.current_version_hits, 
//               100.0 * all_stats.current_hit_rate);
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
