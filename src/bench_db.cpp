#include <fmt/core.h>

#include "common.hpp"

#include <boost/unordered/unordered_flat_map.hpp>

#define DBKIND 0    // custom
// #define DBKIND 1 // leveldb


#if defined(DBKIND) && DBKIND == 1
#include "leveldb_v1.hpp"
using utxo_db = utxo::utxo_db_leveldb;
#elif defined(DBKIND) && DBKIND == 0
#include "interprocess_multiple_v6.hpp"
using utxo_db = utxo::utxo_db;
#endif 


using to_insert_utxos_t = boost::unordered_flat_map<utxo_key_t, kth::domain::chain::output>;
// using to_insert_utxos_t = std::vector<std::pain<utxo_key_t, kth::domain::chain::output>>;
using to_delete_utxos_t = boost::unordered_flat_map<utxo_key_t, kth::domain::chain::input>;

bool is_op_return(kth::domain::chain::output const& output, uint32_t height) {
    if (output.script().bytes().empty()) {
        // log_print("Output script is empty at height {}\n", height);
        return false; // Empty script is not OP_RETURN
    }
    return output.script().bytes()[0] == 0x6a; // OP_RETURN
}

std::tuple<to_insert_utxos_t, to_delete_utxos_t, size_t, size_t> process_in_block(std::vector<kth::domain::chain::transaction>& txs, uint32_t height) {

    size_t skipped_op_return = 0;
    to_insert_utxos_t to_insert;
    // using utxo_key_t = std::array<std::uint8_t, utxo_key_size>;
    // the utxo_key_t is 36 bytes, the first 32 bytes are the transaction hash 
    // and the last 4 bytes are the output index

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
                // skip OP_RETURN outputs
                // log_print("Skipping OP_RETURN output in transaction.\n");
                // print_hash(tx_hash);
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
                // this is an input of coinbase transaction, which is not valid
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
    // Create a timestamped filename
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
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

int main(int argc, char** argv) {

    std::string_view const path = "/home/fernando/dev/utxo-experiments/src";
    init_log_file("bench_db_apple_1");

    size_t total_inputs;
    size_t total_outputs;
    size_t partial_inputs;
    size_t partial_outputs;
    size_t height = 0;

    utxo_db db;

    log_print("Opening DB ...\n");
    db.configure("utxo_interprocess_multiple", true); 
    log_print("DB opened with size: {}\n", db.size());

    process(path,
        [&](auto const& tx_hashes, auto&& txs) {
            // log_print("txs.size() = {}\n", txs.size());
            // log_print("do something with txs\n");

            log_print("Processing block with {} transactions...\n", txs.size());
            auto const [
                to_insert, 
                to_delete,
                in_block_utxos_count,
                skipped_op_return
            ] = process_in_block(txs, height);

            log_print("Processed block with {} inputs and {} outputs. Removed in the same block: {}. Skipped OP_RETURNs: {}\n", 
                      to_delete.size(), to_insert.size(), in_block_utxos_count, skipped_op_return);

            log_print("deleting inputs...\n");
            // first, delete the inputs
            for (auto const& [k, v] : to_delete) {
                db.erase(k, height);
            }

            log_print("Inserting outputs...\n");
            // then, insert the outputs
            for (auto const& [k, v] : to_insert) {
                db.insert(k, v.to_data(), height);
            }

            auto deferred = db.deferred_deletions_size();
            if (deferred > 0) {
                log_print("Processing pending deletions... ({} pending)\n", deferred);
                auto [deleted, failed] = db.process_pending_deletions();
                log_print("Deleted {} entries, {} failed, "
                          "{} pending deletions left\n", 
                          deleted, failed.size(), db.deferred_deletions_size()); 

                if (failed.size() > 0) {
                    log_print("Failed to delete {} entries, these are ERRORS\n", failed.size());
                    for (auto const& f : failed) {
                        log_print("Failed to delete: ");
                        utxo::print_key(f);
                    }
                    std::terminate(); // or handle the error as needed
                }
            } 

        },
        [&]() {
            // Imprimir estadísticas después de cada bloque
            log_print("\n=== Post-block Statistics ===\n");
            db.print_statistics();
            
            // O si quieres procesar las estadísticas de otra forma:
            auto stats = db.get_statistics();
            
            // Por ejemplo, puedes guardar las estadísticas en un archivo
            // o hacer análisis específicos
            
            // // Verificar la salud de la base de datos
            // if (stats.deferred.max_queue_size > 10000) {
            //     log_print("WARNING: Deferred deletion queue is getting large!\n");
            // }
            
            // if (stats.cache_hit_rate < 0.5) {
            //     log_print("WARNING: Cache hit rate is low, consider increasing cache size\n");
            // }
            
            // Resetear estadísticas de búsqueda si quieres stats por bloque
            // db.reset_search_stats();
        },
        total_inputs, total_outputs, partial_inputs, partial_outputs);

    log_print("Processing completed.\n");
    db.print_statistics();

    log_print("Closing DB... \n");
    db.close();
    log_print("DB closed ...\n");

    log_print("Total inputs:    {}\n", total_inputs);
    log_print("Total outputs:   {}\n", total_outputs);
    log_print("Partial Inputs:  {:7}\n", partial_inputs);
    log_print("Partial Outputs: {:7}\n", partial_outputs);

    close_log_file();

    return 0;
}
