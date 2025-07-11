#pragma once

#include <leveldb/db.h>
#include <leveldb/options.h>
#include <leveldb/filter_policy.h>
#include <leveldb/cache.h>
#include <leveldb/write_batch.h>

#include <span>
#include <optional>
#include <array>
#include <chrono>
#include <filesystem>
#include <map>
#include <atomic>

#include "log.hpp"
#include "common_db.hpp"
#include "common_utxo.hpp"

namespace utxo {

namespace fs = std::filesystem;


// Utility function to create leveldb::Slice from utxo_key_t
inline leveldb::Slice key_to_slice(utxo_key_t const& key) {
    return leveldb::Slice(reinterpret_cast<char const*>(key.data()), key.size());
}

// LevelDB-based UTXO database implementation
class utxo_db_leveldb {
public:
    // Statistics structures similar to your custom implementation
    struct db_statistics {
        size_t total_entries = 0;
        size_t total_inserts = 0;
        size_t total_deletes = 0;
        size_t successful_finds = 0;
        size_t failed_finds = 0;
        size_t deferred_deletions = 0;
        
        // Performance metrics
        std::chrono::nanoseconds total_insert_time{0};
        std::chrono::nanoseconds total_delete_time{0};
        std::chrono::nanoseconds total_find_time{0};
        
        // LevelDB specific stats
        std::string leveldb_stats_string;
        
        double get_insert_rate() const {
            if (total_insert_time.count() == 0) return 0.0;
            return double(total_inserts) * 1e9 / total_insert_time.count();
        }
        
        double get_delete_rate() const {
            if (total_delete_time.count() == 0) return 0.0;
            return double(total_deletes) * 1e9 / total_delete_time.count();
        }
        
        double get_find_rate() const {
            if (total_find_time.count() == 0) return 0.0;
            return double(successful_finds + failed_finds) * 1e9 / total_find_time.count();
        }
        
        double get_hit_rate() const {
            auto total_finds = successful_finds + failed_finds;
            if (total_finds == 0) return 0.0;
            return double(successful_finds) / total_finds;
        }
    };

    // Constructor
    utxo_db_leveldb() = default;
    
    // Destructor
    ~utxo_db_leveldb() {
        close();
    }
    
    // No copy
    utxo_db_leveldb(utxo_db_leveldb const&) = delete;
    utxo_db_leveldb& operator=(utxo_db_leveldb const&) = delete;
    
    // Move constructor and assignment
    utxo_db_leveldb(utxo_db_leveldb&& other) noexcept 
        : db_(std::exchange(other.db_, nullptr))
        , stats_(std::move(other.stats_))
        , deferred_deletions_(std::move(other.deferred_deletions_))
    {}
    
    utxo_db_leveldb& operator=(utxo_db_leveldb&& other) noexcept {
        if (this != &other) {
            close();
            db_ = std::exchange(other.db_, nullptr);
            stats_ = std::move(other.stats_);
            deferred_deletions_ = std::move(other.deferred_deletions_);
        }
        return *this;
    }

    // Configure and open the database
    bool configure(std::string_view path, bool remove_existing = false) {
        close(); // Close any existing connection
        
        db_path_ = path;
        
        if (remove_existing && fs::exists(path)) {
            fs::remove_all(path);
            log_print("Removed existing database at: {}\n", path);
        }
        
        // Ensure directory exists
        fs::create_directories(path);
        
        leveldb::Options options;
        
        // Tuning for UTXO workload (write-heavy with some reads)
        options.create_if_missing = true;
        options.error_if_exists = false;
        
        // Large write buffer for better write performance (default is 4MB)
        options.write_buffer_size = 64 * 1024 * 1024; // 64MB
        
        // More aggressive file size (default is 2MB)
        options.max_file_size = 64 * 1024 * 1024; // 64MB
        
        // Block cache for reads (default is 8MB)
        options.block_cache = leveldb::NewLRUCache(256 * 1024 * 1024); // 256MB
        
        // Block size (default is 4KB)
        options.block_size = 16 * 1024; // 16KB
        
        // Bloom filter to reduce disk reads for non-existent keys
        options.filter_policy = leveldb::NewBloomFilterPolicy(10);
        
        // Compression (Snappy is usually fastest)
        options.compression = leveldb::kSnappyCompression;
        
        // More aggressive compaction
        options.max_open_files = 1000;
        
        leveldb::Status status = leveldb::DB::Open(options, std::string(path), &db_);
        
        if (!status.ok()) {
            log_print("Failed to open LevelDB: {}\n", status.ToString());
            return false;
        }
        
        log_print("LevelDB opened successfully at: {}\n", path);
        
        // Reset statistics
        stats_ = db_statistics{};
        deferred_deletions_.clear();
        
        return true;
    }
    
    // Close the database
    void close() {
        if (db_) {
            delete db_;
            db_ = nullptr;
            log_print("LevelDB closed\n");
        }
    }
    
    // Get current size (approximate)
    size_t size() const {
        return stats_.total_entries;
    }
    
    // Insert a key-value pair
    bool insert(utxo_key_t const& key, span_bytes value, uint32_t height) {
        if (!db_) {
            log_print("insert: Database not open\n");
            return false;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Create value with height prefix (similar to your implementation)
        std::vector<uint8_t> full_value;
        full_value.reserve(sizeof(uint32_t) + value.size());
        
        // Add height as prefix
        uint8_t const* height_bytes = reinterpret_cast<uint8_t const*>(&height);
        full_value.insert(full_value.end(), height_bytes, height_bytes + sizeof(uint32_t));
        
        // Add actual value
        full_value.insert(full_value.end(), value.begin(), value.end());
        
        leveldb::WriteOptions write_options;
        write_options.sync = false; // Don't force fsync for better performance
        
        leveldb::Status status = db_->Put(
            write_options,
            key_to_slice(key),
            leveldb::Slice(reinterpret_cast<char const*>(full_value.data()), full_value.size())
        );
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        stats_.total_insert_time += duration;
        
        if (status.ok()) {
            ++stats_.total_inserts;
            ++stats_.total_entries;
            return true;
        } else {
            log_print("insert: LevelDB Put failed: {}\n", status.ToString());
            return false;
        }
    }
    
    // Delete a key (erase)
    size_t erase(utxo_key_t const& key, uint32_t height) {
        if (!db_) {
            log_print("erase: Database not open\n");
            return 0;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // First check if key exists
        std::string existing_value;
        leveldb::ReadOptions read_options;
        leveldb::Status read_status = db_->Get(read_options, key_to_slice(key), &existing_value);
        
        if (!read_status.ok()) {
            // Key doesn't exist, add to deferred deletions for compatibility
            add_to_deferred_deletions(key, height);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            stats_.total_delete_time += duration;
            return 0;
        }
        
        // Delete the key
        leveldb::WriteOptions write_options;
        write_options.sync = false;
        
        leveldb::Status delete_status = db_->Delete(write_options, key_to_slice(key));
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        stats_.total_delete_time += duration;
        
        if (delete_status.ok()) {
            ++stats_.total_deletes;
            --stats_.total_entries;
            return 1;
        } else {
            log_print("erase: LevelDB Delete failed: {}\n", delete_status.ToString());
            return 0;
        }
    }
    
    // Find a key
    std::optional<std::vector<uint8_t>> find(utxo_key_t const& key, uint32_t height) {
        if (!db_) {
            log_print("find: Database not open\n");
            return std::nullopt;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        std::string value;
        leveldb::ReadOptions read_options;
        leveldb::Status status = db_->Get(read_options, key_to_slice(key), &value);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        stats_.total_find_time += duration;
        
        if (status.ok()) {
            ++stats_.successful_finds;
            
            // Extract the actual value (skip height prefix)
            if (value.size() >= sizeof(uint32_t)) {
                std::vector<uint8_t> result;
                result.reserve(value.size() - sizeof(uint32_t));
                
                uint8_t const* data = reinterpret_cast<uint8_t const*>(value.data());
                result.insert(result.end(), 
                             data + sizeof(uint32_t), 
                             data + value.size());
                
                return result;
            }
        }
        
        ++stats_.failed_finds;
        return std::nullopt;
    }
    
    // Get statistics
    db_statistics get_statistics() {
        // Update LevelDB internal stats
        if (db_) {
            std::string stats_string;
            if (db_->GetProperty("leveldb.stats", &stats_string)) {
                stats_.leveldb_stats_string = std::move(stats_string);
            }
        }
        
        return stats_;
    }
    
    // Print statistics
    void print_statistics() {
        auto stats = get_statistics();
        
        log_print("\n=== LevelDB UTXO Database Statistics ===\n");
        log_print("Total entries: {}\n", stats.total_entries);
        log_print("Total inserts: {}\n", stats.total_inserts);
        log_print("Total deletes: {}\n", stats.total_deletes);
        log_print("Successful finds: {}\n", stats.successful_finds);
        log_print("Failed finds: {}\n", stats.failed_finds);
        log_print("Deferred deletions: {}\n", stats.deferred_deletions);
        
        log_print("\n--- Performance Metrics ---\n");
        log_print("Insert rate: {:.2f} ops/sec\n", stats.get_insert_rate());
        log_print("Delete rate: {:.2f} ops/sec\n", stats.get_delete_rate());
        log_print("Find rate: {:.2f} ops/sec\n", stats.get_find_rate());
        log_print("Hit rate: {:.2f}%\n", stats.get_hit_rate() * 100.0);
        
        if (!stats.leveldb_stats_string.empty()) {
            log_print("\n--- LevelDB Internal Stats ---\n");
            log_print("{}\n", stats.leveldb_stats_string);
        }
        
        log_print("================================\n");
    }
    
    // Deferred deletions interface (for compatibility)
    size_t deferred_deletions_size() const {
        return deferred_deletions_.size();
    }
    
    // Process deferred deletions
    std::pair<uint32_t, std::vector<utxo_key_t>> process_pending_deletions() {
        std::vector<utxo_key_t> failed_deletions;
        uint32_t successful_deletions = 0;
        
        if (!db_) {
            log_print("process_pending_deletions: Database not open\n");
            return {0, std::move(failed_deletions)};
        }
        
        leveldb::WriteBatch batch;
        
        for (auto it = deferred_deletions_.begin(); it != deferred_deletions_.end(); ) {
            utxo_key_t const& key = it->first;
            
            // Check if key exists before trying to delete
            std::string value;
            leveldb::ReadOptions read_options;
            leveldb::Status read_status = db_->Get(read_options, key_to_slice(key), &value);
            
            if (read_status.ok()) {
                // Key exists, add to batch delete
                batch.Delete(key_to_slice(key));
                ++successful_deletions;
                it = deferred_deletions_.erase(it);
            } else {
                // Key doesn't exist, keep in deferred list or mark as failed
                failed_deletions.push_back(key);
                it = deferred_deletions_.erase(it);
            }
        }
        
        // Execute batch delete
        if (successful_deletions > 0) {
            leveldb::WriteOptions write_options;
            write_options.sync = false;
            
            leveldb::Status batch_status = db_->Write(write_options, &batch);
            if (!batch_status.ok()) {
                log_print("process_pending_deletions: Batch write failed: {}\n", batch_status.ToString());
                // In case of failure, we might want to re-add items to deferred list
            } else {
                stats_.total_deletes += successful_deletions;
                stats_.total_entries -= successful_deletions;
            }
        }
        
        return {successful_deletions, std::move(failed_deletions)};
    }
    
    // Reset statistics
    void reset_all_statistics() {
        stats_ = db_statistics{};
    }
    
    // Compact database (useful for performance testing)
    void compact() {
        if (db_) {
            log_print("Compacting LevelDB...\n");
            db_->CompactRange(nullptr, nullptr);
            log_print("LevelDB compaction completed\n");
        }
    }
    
    // Get approximate database size on disk
    uint64_t get_database_size() const {
        if (!fs::exists(db_path_)) {
            return 0;
        }
        
        uint64_t total_size = 0;
        std::error_code ec;
        
        for (auto const& entry : fs::recursive_directory_iterator(db_path_, ec)) {
            if (!ec && entry.is_regular_file()) {
                total_size += entry.file_size(ec);
            }
        }
        
        return total_size;
    }

private:
    leveldb::DB* db_ = nullptr;
    fs::path db_path_;
    db_statistics stats_;
    
    // Deferred deletions map (key -> height)
    std::map<utxo_key_t, uint32_t> deferred_deletions_;
    
    // Helper to add key to deferred deletions
    void add_to_deferred_deletions(utxo_key_t const& key, uint32_t height) {
        deferred_deletions_[key] = height;
        ++stats_.deferred_deletions;
    }
};

} // namespace utxo
