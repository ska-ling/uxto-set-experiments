#include <fmt/core.h>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "common.hpp"

#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>

#define DBKIND 0    // custom
// #define DBKIND 1 // leveldb

#if defined(DBKIND) && DBKIND == 1
#include "leveldb_v1.hpp"
using utxo_db = utxo::utxo_db_leveldb;
#elif defined(DBKIND) && DBKIND == 0
// #include "interprocess_multiple_v8.hpp"
// #include "interprocess_multiple_v9.hpp"

// Line without index files
// #include "interprocess_multiple_v6.hpp"
// #include "interprocess_multiple_v10.hpp"
// #include "interprocess_multiple_v11.hpp" // compactación de archivos de datos.
#include "interprocess_multiple_v12.hpp" // sin logica de op return

using utxo_db = utxo::utxo_db;
#endif 

// Configuration constants - parameterized for easy modification
constexpr size_t SYNC_UP_TO_BLOCK = 750'000;      // Block height to sync up to (DB should be pre-synced)
constexpr size_t LOOKUP_FROM_BLOCK = 750'000;     // Start reading transactions from this block
constexpr size_t LOOKUP_TO_BLOCK = 780'000;       // End reading transactions at this block

// constexpr size_t MIN_LOOKUPS_PER_BATCH = 50'000;  // Minimum lookups per batch
// constexpr size_t MAX_LOOKUPS_PER_BATCH = 100'000; // Maximum lookups per batch

constexpr size_t MIN_LOOKUPS_PER_BATCH = 500'000;  // Minimum lookups per batch
constexpr size_t MAX_LOOKUPS_PER_BATCH = 1'000'000; // Maximum lookups per batch

// constexpr double FAILED_LOOKUP_RATIO = 0.1;       // 10% of lookups should be known failures
constexpr double FAILED_LOOKUP_RATIO = 0.0;       //  0% of lookups should be known failures
constexpr size_t MIN_THREADS = 1;                 // Minimum number of threads to test
constexpr size_t MAX_THREADS = 64;                // Maximum number of threads to test
constexpr bool SHUFFLE_LOOKUP_ITEMS = false;       // Whether to shuffle lookup items or keep blockchain order

struct LookupItem {
    utxo_key_t key;
    bool should_exist;  // true = should be found, false = should fail
};

using LookupBatch = std::vector<LookupItem>;

// Results from processing lookup transactions
struct LookupPreprocessResult {
    LookupBatch lookup_items;
    size_t total_outputs_generated;
    size_t outputs_filtered_out;
    size_t failed_lookups = 0; // Count of lookups that should fail
};

// Generate a set of outputs that will be created by transactions in the given transaction vector
// These should be filtered out from our lookup tests since they won't exist in the pre-synced DB
boost::unordered_flat_set<utxo_key_t> get_outputs_from_transactions(
    const std::vector<kth::domain::chain::transaction>& transactions) {
    
    boost::unordered_flat_set<utxo_key_t> outputs_in_range;
    
    log_print("Collecting outputs from {} transactions...\n", transactions.size());
    
    for (auto const& tx : transactions) {
        auto tx_hash = tx.hash();
        utxo_key_t current_key;
        std::copy(tx_hash.begin(), tx_hash.end(), current_key.begin());
        
        size_t output_index = 0;
        for (auto const& output : tx.outputs()) {
            // Copy output index into the key
            std::copy(reinterpret_cast<const uint8_t*>(&output_index),
                     reinterpret_cast<const uint8_t*>(&output_index) + 4,
                     current_key.end() - 4);
            
            outputs_in_range.insert(current_key);
            ++output_index;
        }
    }
    
    log_print("Collected {} outputs from {} transactions\n",
              outputs_in_range.size(), transactions.size());
    
    return outputs_in_range;
}

// Preprocess transactions to generate lookup test cases
LookupPreprocessResult preprocess_for_lookups(
    std::filesystem::path const& path,
    size_t target_lookups,
    size_t from_block,
    size_t to_block) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> failed_lookup_dist(0, int(1.0 / FAILED_LOOKUP_RATIO) - 1); // For 0.1 ratio: 0-9, where 0 means failed lookup
    
    LookupBatch lookup_items;
    lookup_items.reserve(target_lookups);
    
    size_t total_outputs_generated = 0;
    size_t outputs_filtered_out = 0;
    size_t failed_lookups = 0; // Count of lookups that should fail
    
    log_print("Preprocessing transactions for lookup tests...\n");
    log_print("Target lookups: {}, Range: [{}, {})\n", target_lookups, from_block, to_block);
    
    // Read all transactions from the range in one pass
    size_t current_block = from_block;
    size_t current_tx = 0;
    size_t estimated_transactions_needed = target_lookups * 2; // Estimate to get enough data
    
    log_print("Reading transactions from files (estimated need: {})...\n", estimated_transactions_needed);
    auto [transactions, next_block, next_tx, stop] = get_n_transactions(path, current_block, current_tx, estimated_transactions_needed);
    
    if (transactions.empty()) {
        log_print("No transactions found in range [{}, {})\n", from_block, to_block);
        return {std::move(lookup_items), 0, 0};
    }
    
    log_print("Read {} transactions, now processing for lookup generation...\n", transactions.size());
    
    // First pass: collect all outputs that will be generated by these transactions
    // These should be filtered out since they won't exist in the pre-synced DB
    auto outputs_in_range = get_outputs_from_transactions(transactions);
    
    // Second pass: process inputs for lookup tests
    for (auto const& tx : transactions) {
        for (auto const& input : tx.inputs()) {
            if (input.previous_output().is_null()) {
                continue; // Skip coinbase inputs
            }
            
            auto const& prev_out = input.previous_output();
            auto const& hash = prev_out.hash();
            auto index = prev_out.index();
            
            utxo_key_t key;
            std::copy(hash.begin(), hash.end(), key.begin());
            std::copy(reinterpret_cast<const uint8_t*>(&index),
                     reinterpret_cast<const uint8_t*>(&index) + 4,
                     key.end() - 4);
            
            total_outputs_generated++;
            
            // Filter out outputs that are created in our lookup range
            if (outputs_in_range.contains(key)) {
                // Use this filtered output as a failed lookup case if the random generator decides it
                if (failed_lookup_dist(gen) == 0) {
                    // This key won't exist in the pre-synced DB, perfect for a failed lookup test
                    lookup_items.push_back({key, false});
                    ++failed_lookups;
                } else {
                    ++outputs_filtered_out;
                }
            } else {
                // This should exist in the pre-synced DB
                lookup_items.push_back({key, true});
            }
            
            
            if (lookup_items.size() >= target_lookups) {
                break;
            }
        }
        
        if (lookup_items.size() >= target_lookups) {
            break;
        }
    }
    
    // Optionally shuffle the lookup items to mix successful and failed lookups
    if (SHUFFLE_LOOKUP_ITEMS) {
        std::shuffle(lookup_items.begin(), lookup_items.end(), gen);
        log_print("Lookup items shuffled\n");
    } else {
        log_print("Lookup items kept in blockchain order\n");
    }
    
    log_print("Preprocessing complete:\n");
    log_print("  Total lookup items: {}\n", lookup_items.size());
    log_print("  Expected successes: {}\n", std::count_if(lookup_items.begin(), lookup_items.end(), 
                                                          [](const auto& item) { return item.should_exist; }));
    log_print("  Expected failures: {}\n", std::count_if(lookup_items.begin(), lookup_items.end(), 
                                                         [](const auto& item) { return !item.should_exist; }));
    log_print("  Total outputs examined: {}\n", total_outputs_generated);
    log_print("  Outputs filtered out: {}\n", outputs_filtered_out);
    log_print("  Failed lookups (should not exist): {}\n", failed_lookups);
    
    return {std::move(lookup_items), total_outputs_generated, outputs_filtered_out, failed_lookups};
}

// Perform lookup benchmarks with specified number of threads
struct LookupBenchmarkResult {
    size_t thread_count;
    size_t total_lookups;
    size_t successful_lookups;
    size_t failed_lookups;
    size_t unexpected_results;  // Found when shouldn't exist, or not found when should exist
    double total_time_ns;
    double direct_lookup_time_ns;  // Time for direct lookups (excluding deferred processing)
    double deferred_time_ns;       // Time spent processing deferred lookups
    size_t deferred_count;         // Number of deferred lookups processed
    double lookups_per_second;
};

LookupBenchmarkResult benchmark_lookups(utxo_db& db, const LookupBatch& lookup_items, size_t thread_count) {
    log_print("Starting lookup benchmark with {} threads, {} items...\n", thread_count, lookup_items.size());
    
    std::atomic<size_t> successful_lookups{0};
    std::atomic<size_t> failed_lookups{0};
    std::atomic<size_t> unexpected_results{0};
    
    size_t items_per_thread = lookup_items.size() / thread_count;
    size_t remaining_items = lookup_items.size() % thread_count;
    
    // Define the worker lambda outside the thread creation
    auto worker = [&db, &lookup_items, &successful_lookups, &failed_lookups, &unexpected_results](size_t start_idx, size_t end_idx) {
        for (size_t i = start_idx; i < end_idx; ++i) {
            const auto& item = lookup_items[i];
            auto result = db.find(item.key, 0); // height parameter not used for lookups
            if (result) {
                // Found directly in current version or cached files
                successful_lookups.fetch_add(1, std::memory_order_relaxed);
            } 
            // If not found, it might be added to deferred lookups
            // We'll process the final results after deferred processing
        }
    };
    
    auto const start_time = std::chrono::high_resolution_clock::now();
    
    if (thread_count == 1) {
        // Single-threaded execution in main thread
        worker(0, lookup_items.size());
    } else {
        // Multi-threaded execution: create N-1 threads, main thread does the last chunk
        std::vector<std::thread> threads;
        threads.reserve(thread_count - 1);
        
        // Create N-1 threads
        for (size_t t = 0; t < thread_count - 1; ++t) {
            size_t start_idx = t * items_per_thread;
            size_t end_idx = start_idx + items_per_thread;
            
            threads.emplace_back(worker, start_idx, end_idx);
        }
        
        // Main thread processes the last chunk (including any remaining items)
        size_t main_start_idx = (thread_count - 1) * items_per_thread;
        size_t main_end_idx = lookup_items.size(); // This includes remaining_items
        worker(main_start_idx, main_end_idx);
        
        // Wait for all worker threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
    auto const end_time = std::chrono::high_resolution_clock::now();
    double direct_lookup_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    double total_time_ns = direct_lookup_time_ns;
    double deferred_time_ns = 0.0;
    size_t deferred_count = 0;

#if defined(DBKIND) && DBKIND == 0
    // Process deferred lookups if any (only for our custom DB)
    deferred_count = db.deferred_lookups_size();
    log_print("After direct lookups: {} successful direct hits, {} deferred lookups queued\n", 
              successful_lookups.load(), deferred_count);
    
    if (deferred_count > 0) {
        log_print("Processing {} deferred lookups...\n", deferred_count);
        
        auto const deferred_start = std::chrono::high_resolution_clock::now();
        auto [successful_deferred, failed_deferred] = db.process_pending_lookups();
        auto const deferred_end = std::chrono::high_resolution_clock::now();
        
        deferred_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(deferred_end - deferred_start).count();
        total_time_ns += deferred_time_ns; // Add deferred processing time to total
        
        log_print("Deferred lookups processed:\n");
        log_print("  Successful: {}\n", successful_deferred.size());
        log_print("  Failed: {}\n", failed_deferred.size());
        log_print("  Deferred processing time: {}\n", format_time(deferred_time_ns));
        log_print("  Deferred processing rate: {}/sec\n", format_si_rate((successful_deferred.size() + failed_deferred.size()) * 1e9 / deferred_time_ns));
        
        // Update counters with deferred results
        successful_lookups.fetch_add(successful_deferred.size(), std::memory_order_relaxed);
        
        // Calculate total failed lookups (items not found in either direct or deferred lookups)
        size_t total_found = successful_lookups.load();
        size_t total_not_found = lookup_items.size() - total_found;
        failed_lookups.store(total_not_found, std::memory_order_relaxed);
        
        // Verify the deferred queue is now empty
        size_t final_deferred_size = db.deferred_lookups_size();
        if (final_deferred_size > 0) {
            log_print("ERROR: Deferred lookups queue not empty after processing! Size: {}\n", final_deferred_size);
            std::terminate();
        }
        log_print("Deferred lookup queue successfully cleared (final size: {})\n", final_deferred_size);
        
        // Note: We don't update unexpected_results for deferred lookups since we don't track
        // their expected outcomes separately
    } else {
        // No deferred lookups to process - calculate failed lookups directly
        size_t total_found = successful_lookups.load();
        size_t total_not_found = lookup_items.size() - total_found;
        failed_lookups.store(total_not_found, std::memory_order_relaxed);
        log_print("No deferred lookups to process. Direct results: {} found, {} not found\n",
                 total_found, total_not_found);
    }
#else
    // For LevelDB or when custom DB is not being used
    // Calculate failed lookups directly
    size_t total_found = successful_lookups.load();
    size_t total_not_found = lookup_items.size() - total_found;
    failed_lookups.store(total_not_found, std::memory_order_relaxed);
#endif
    
    double lookups_per_second = (lookup_items.size() * 1e9) / total_time_ns;
    
    return {
        thread_count,
        lookup_items.size(),
        successful_lookups.load(),
        failed_lookups.load(),
        unexpected_results.load(),
        total_time_ns,
        direct_lookup_time_ns,
        deferred_time_ns,
        deferred_count,
        lookups_per_second
    };
}

// Global log file - define the variable declared in log.hpp
std::ofstream log_file;

void init_log_file(const std::string& benchmark_name);
void close_log_file();

// Initialize the log file with a timestamped name
void init_log_file(const std::string& benchmark_name) {
    // Create a timestamped filename
    auto const now = std::chrono::system_clock::now();
    auto const now_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "benchmark_" << benchmark_name << "_" 
       << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S") << ".log";
    
    log_file.open(ss.str());
    if (!log_file.is_open()) {
        std::cerr << "Failed to open log file: " << ss.str() << std::endl;
    } else {
        // Format the timestamp as string first
        std::stringstream time_ss;
        time_ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S");
        std::string timestamp = time_ss.str();
        
        // Now use the string with log_print
        log_print("Log started at: {}\n", timestamp);
        log_print("Benchmark: {}\n\n", benchmark_name);
    }
}

// Close the log file
void close_log_file() {
    if (log_file.is_open()) {
        log_print("Log completed.\n");
        log_file.close();
    }
}

int main() {
    using namespace std::filesystem;
    
    init_log_file("benchmark_lookup");
    
    log_print("=== UTXO Lookup Benchmark ===\n");
    log_print("Configuration:\n");
    log_print("  Sync up to block: {}\n", SYNC_UP_TO_BLOCK);
    log_print("  Lookup range: [{}, {})\n", LOOKUP_FROM_BLOCK, LOOKUP_TO_BLOCK);
    log_print("  Lookups per batch: [{}, {}]\n", MIN_LOOKUPS_PER_BATCH, MAX_LOOKUPS_PER_BATCH);
    log_print("  Failed lookup ratio: {:.1f}%\n", FAILED_LOOKUP_RATIO * 100.0);
    log_print("  Thread range: [{}, {}]\n", MIN_THREADS, MAX_THREADS);
    log_print("  Shuffle lookup items: {}\n", SHUFFLE_LOOKUP_ITEMS ? "Yes" : "No");
    
    // Initialize database (should be pre-synced to SYNC_UP_TO_BLOCK)
    utxo_db db;
    
#if defined(DBKIND) && DBKIND == 1
    db.configure("./leveldb_utxos", false); // read-only mode (don't remove existing)
    log_print("Database opened successfully (LevelDB)\n");
#elif defined(DBKIND) && DBKIND == 0
    db.configure("utxo_interprocess_multiple", false); // read-only mode (don't remove existing)
    log_print("Database opened successfully (Custom DB)\n");
#endif
    
    // Get data path
    path const blocks_path = "/home/fernando/dev/utxo-experiments/src";
    if (!exists(blocks_path)) {
        log_print("Error: blocks path does not exist: {}\n", blocks_path);
        close_log_file();
        return 1;
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> lookup_dist(MIN_LOOKUPS_PER_BATCH, MAX_LOOKUPS_PER_BATCH);
    
    // Run multiple benchmark iterations
    std::vector<LookupBenchmarkResult> all_results;
    
    for (int iteration = 0; iteration < 5; ++iteration) {
        log_print("\n=== Iteration {} ===\n", iteration + 1);
        
        // Generate random number of lookups for this iteration
        size_t target_lookups = lookup_dist(gen);
        
        // Preprocess transactions to get lookup test cases
        auto preprocess_result = preprocess_for_lookups(
            blocks_path, 
            target_lookups, 
            LOOKUP_FROM_BLOCK, 
            LOOKUP_TO_BLOCK
        );
        
        if (preprocess_result.lookup_items.empty()) {
            log_print("No lookup items generated, skipping iteration\n");
            continue;
        }
        
        // Test with different thread counts
        for (size_t thread_count = MIN_THREADS; thread_count <= MAX_THREADS; thread_count *= 2) {
            auto result = benchmark_lookups(db, preprocess_result.lookup_items, thread_count);
            all_results.push_back(result);
            
            // Strict validation: Check if expected failed lookups match actual failed lookups
            size_t expected_failed = preprocess_result.failed_lookups;
            size_t actual_failed = result.failed_lookups;
            
            log_print("\nValidation Results (Iteration {}, {} threads):\n", iteration + 1, thread_count);
            log_print("  Expected failed lookups: {}\n", expected_failed);
            log_print("  Actual failed lookups: {}\n", actual_failed);
            
            if (expected_failed != actual_failed) {
                log_print("ERROR: Expected failed lookups ({}) != Actual failed lookups ({})\n", 
                        expected_failed, actual_failed);
                log_print("This indicates a serious issue with the benchmark logic or database state!\n");
                std::terminate();
            } else {
                log_print("✓ Validation passed: Expected and actual failed lookups match\n");
            }
            
            log_print("\nBenchmark Results (Iteration {}, {} threads):\n", iteration + 1, thread_count);
            log_print("  Total lookups: {}\n", format_si(result.total_lookups));
            log_print("  Successful: {}\n", format_si(result.successful_lookups));
            log_print("  Failed: {}\n", format_si(result.failed_lookups));
            log_print("  Direct lookup time: {}\n", format_time(result.direct_lookup_time_ns));
            if (result.deferred_count > 0) {
                log_print("  Deferred count: {}\n", format_si(result.deferred_count));
                log_print("  Deferred time: {}\n", format_time(result.deferred_time_ns));
            }
            log_print("  Total time: {}\n", format_time(result.total_time_ns));
            log_print("  Overall rate: {}/sec\n", format_si_rate(result.lookups_per_second));
            if (result.deferred_count > 0) {
                double direct_rate = (result.total_lookups - result.deferred_count) * 1e9 / result.direct_lookup_time_ns;
                log_print("  Direct lookup rate: {}/sec\n", format_si_rate(direct_rate));
            }
        }
    }
    
    // Print summary of all results
    log_print("\n=== Summary of All Results ===\n");
    log_print("{:>7} {:>10} {:>12} {:>12} {:>10} {:>15} {:>15}\n", 
              "Threads", "Lookups", "Successful", "Failed", "Deferred", "Total/sec", "Direct/sec");
    log_print("{:-^7} {:-^10} {:-^12} {:-^12} {:-^10} {:-^15} {:-^15}\n", "", "", "", "", "", "", "");
    
    for (const auto& result : all_results) {
        double direct_rate = result.deferred_count > 0 ? 
            ((result.total_lookups - result.deferred_count) * 1e9 / result.direct_lookup_time_ns) : 
            result.lookups_per_second;
            
        log_print("{:>7} {:>10} {:>12} {:>12} {:>10} {:>15} {:>15}\n",
                  result.thread_count,
                  format_si(result.total_lookups),
                  format_si(result.successful_lookups), 
                  format_si(result.failed_lookups),
                  format_si(result.deferred_count),
                  format_si_rate(result.lookups_per_second),
                  format_si_rate(direct_rate));
    }
    
    // Analysis: find optimal thread count
    log_print("\n=== Performance Analysis ===\n");
    std::map<size_t, std::vector<double>> performance_by_threads;
    
    for (const auto& result : all_results) {
        performance_by_threads[result.thread_count].push_back(result.lookups_per_second);
    }
    
    for (const auto& [thread_count, performances] : performance_by_threads) {
        double avg_performance = std::accumulate(performances.begin(), performances.end(), 0.0) / performances.size();
        double max_performance = *std::max_element(performances.begin(), performances.end());
        double min_performance = *std::min_element(performances.begin(), performances.end());
        
        log_print("  {} threads: avg={}, min={}, max={}\n",
                  thread_count,
                  format_si_rate(avg_performance),
                  format_si_rate(min_performance), 
                  format_si_rate(max_performance));
    }
    
    log_print("\nLookup benchmark completed.\n");
    db.close();
    close_log_file();
    
    return 0;
}
