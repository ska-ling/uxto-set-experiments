#include <fmt/core.h>

#include <utxo/common.hpp>

// Include both implementations
#include "interprocess_multiple_v6.hpp"
#include "leveldb_v1.hpp"

using to_insert_utxos_t = boost::unordered_flat_map<utxo_key_t, kth::domain::chain::output>;
using to_delete_utxos_t = boost::unordered_flat_map<utxo_key_t, kth::domain::chain::input>;

bool is_op_return(kth::domain::chain::output const& output, uint32_t height) {
    if (output.script().bytes().empty()) {
        return false; // Empty script is not OP_RETURN
    }
    return output.script().bytes()[0] == 0x6a; // OP_RETURN
}

std::tuple<to_insert_utxos_t, to_delete_utxos_t, size_t, size_t> process_in_block(std::vector<kth::domain::chain::transaction>& txs, uint32_t height) {
    size_t skipped_op_return = 0;
    to_insert_utxos_t to_insert;
    
    // insert all the outputs
    for (auto const& tx : txs) {
        auto tx_hash = tx.hash();
        utxo_key_t key;
        // copy the transaction hash into the key
        std::copy(tx_hash.begin(), tx_hash.end(), key.begin());

        size_t output_index = 0;
        for (auto&& output : tx.outputs()) {
            if (is_op_return(output, height)) {
                ++skipped_op_return;
                continue;
            }

            // copy the output index into the key
            std::copy(reinterpret_cast<const uint8_t*>(&output_index), 
                      reinterpret_cast<const uint8_t*>(&output_index) + 4, 
                      key.end() - 4);
            ++output_index;
            to_insert.emplace(std::move(key), std::move(output));
        }
    }

    size_t in_block_utxos = 0;
    to_delete_utxos_t to_delete;

    // remove the inputs
    for (auto const& tx : txs) {
        for (auto&& input : tx.inputs()) {
            auto const& prev_out = input.previous_output();
            auto const& hash = prev_out.hash();
            auto const idx = prev_out.index();
            // if idx == max_uint32, then the input is invalid
            if (idx == std::numeric_limits<uint32_t>::max()) {
                continue; // skip invalid inputs
            }

            utxo_key_t key;
            // copy the transaction hash into the key
            std::copy(hash.begin(), hash.end(), key.begin());

            // copy the output index into the key
            std::copy(reinterpret_cast<const uint8_t*>(&idx), 
                      reinterpret_cast<const uint8_t*>(&idx) + 4, 
                      key.end() - 4);

            // erase the input from the map
            auto const removed = to_insert.erase(key);
            if (removed == 0) {
                to_delete.emplace(std::move(key), std::move(input));
            }
            in_block_utxos += removed;
        }
    }

    return {
        std::move(to_insert), 
        std::move(to_delete),
        in_block_utxos,
        skipped_op_return
    };
}

std::ofstream log_file;

// Initialize the log file
void init_log_file(const std::string& benchmark_name) {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "benchmark_" << benchmark_name << "_" 
       << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S") << ".log";
    
    log_file.open(ss.str());
    if (!log_file.is_open()) {
        std::cerr << "Failed to open log file: " << ss.str() << std::endl;
    } else {
        std::stringstream time_ss;
        time_ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S");
        std::string timestamp = time_ss.str();
        
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

template<typename DB>
void run_benchmark(DB& db, std::string_view db_name, std::string_view path) {
    log_print("\n=== Starting {} Benchmark ===\n", db_name);
    
    size_t total_inputs = 0;
    size_t total_outputs = 0;
    size_t partial_inputs = 0;
    size_t partial_outputs = 0;
    size_t height = 0;
    
    auto benchmark_start = std::chrono::high_resolution_clock::now();
    
    process(path,
        [&](auto const& tx_hashes, auto&& txs) {
            log_print("Processing block with {} transactions...\n", txs.size());
            auto const [
                to_insert, 
                to_delete,
                in_block_utxos_count,
                skipped_op_return
            ] = process_in_block(txs, height);

            log_print("Processed block with {} inputs and {} outputs. Removed in the same block: {}. Skipped OP_RETURNs: {}\n", 
                      to_delete.size(), to_insert.size(), in_block_utxos_count, skipped_op_return);

            // Timing deletions
            auto delete_start = std::chrono::high_resolution_clock::now();
            for (auto const& [k, v] : to_delete) {
                db.erase(k, height);
            }
            auto delete_end = std::chrono::high_resolution_clock::now();
            auto delete_time = std::chrono::duration_cast<std::chrono::milliseconds>(delete_end - delete_start);

            // Timing insertions
            auto insert_start = std::chrono::high_resolution_clock::now();
            for (auto const& [k, v] : to_insert) {
                db.insert(k, v.to_data(), height);
            }
            auto insert_end = std::chrono::high_resolution_clock::now();
            auto insert_time = std::chrono::duration_cast<std::chrono::milliseconds>(insert_end - insert_start);

            log_print("Delete time: {}ms, Insert time: {}ms\n", delete_time.count(), insert_time.count());

            // Process deferred deletions if supported
            if constexpr (requires { db.deferred_deletions_size(); }) {
                auto deferred = db.deferred_deletions_size();
                if (deferred > 0) {
                    log_print("Processing pending deletions... ({} pending)\n", deferred);
                    auto [deleted, failed] = db.process_pending_deletions();
                    log_print("Deleted {} entries, {} failed, "
                              "{} pending deletions left\n", 
                              deleted, failed.size(), db.deferred_deletions_size()); 
                }
            }

            ++height;
        },
        [&]() {
            // Post-processing after each block
            log_print("\n=== {} Post-block Statistics ===\n", db_name);
            db.print_statistics();
        },
        total_inputs, total_outputs, partial_inputs, partial_outputs);

    auto benchmark_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(benchmark_end - benchmark_start);

    log_print("\n=== {} Final Results ===\n", db_name);
    log_print("Total benchmark time: {}ms\n", total_time.count());
    log_print("Total inputs:    {}\n", total_inputs);
    log_print("Total outputs:   {}\n", total_outputs);
    log_print("Partial Inputs:  {:7}\n", partial_inputs);
    log_print("Partial Outputs: {:7}\n", partial_outputs);
    
    db.print_statistics();
    
    // Database-specific final operations
    if constexpr (requires { db.get_database_size(); }) {
        auto db_size = db.get_database_size();
        log_print("Database size on disk: {} bytes ({:.2f} MB)\n", 
                  db_size, double(db_size) / (1024.0 * 1024.0));
    }
    
    log_print("=== {} Benchmark Complete ===\n\n", db_name);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data_path> [db_type]\n";
        std::cerr << "db_type: 'custom', 'leveldb', or 'both' (default: 'both')\n";
        return 1;
    }
    
    std::string_view const path = argv[1];
    std::string_view const db_type = argc >= 3 ? argv[2] : "both";
    
    init_log_file("db_comparison");
    
    log_print("Database Comparison Benchmark\n");
    log_print("Data path: {}\n", path);
    log_print("Database type: {}\n\n", db_type);
    
    try {
        if (db_type == "custom" || db_type == "both") {
            // Test custom implementation
            utxo::utxo_db custom_db;
            log_print("Opening Custom DB...\n");
            custom_db.configure("utxo_interprocess_multiple_comparison", true);
            log_print("Custom DB opened with size: {}\n", custom_db.size());
            
            run_benchmark(custom_db, "Custom DB", path);
            
            log_print("Closing Custom DB...\n");
            custom_db.close();
            log_print("Custom DB closed.\n");
        }
        
        if (db_type == "leveldb" || db_type == "both") {
            // Test LevelDB implementation
            utxo::utxo_db_leveldb leveldb;
            log_print("Opening LevelDB...\n");
            bool success = leveldb.configure("utxo_leveldb_comparison", true);
            if (!success) {
                log_print("Failed to open LevelDB. Skipping LevelDB benchmark.\n");
            } else {
                log_print("LevelDB opened with size: {}\n", leveldb.size());
                
                run_benchmark(leveldb, "LevelDB", path);
                
                log_print("Closing LevelDB...\n");
                leveldb.close();
                log_print("LevelDB closed.\n");
            }
        }
        
    } catch (std::exception const& e) {
        log_print("Error during benchmark: {}\n", e.what());
        close_log_file();
        return 1;
    }
    
    log_print("All benchmarks completed successfully.\n");
    close_log_file();
    return 0;
}
