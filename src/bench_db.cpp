#include <fmt/core.h>

#include <utxo/common.hpp>

// #include "interprocess_multiple_v2.hpp"
#include "interprocess_multiple_v4.hpp"

using to_insert_utxos_t = boost::unordered_flat_map<utxo_key_t, kth::domain::chain::output>;
// using to_insert_utxos_t = std::vector<std::pain<utxo_key_t, kth::domain::chain::output>>;
using to_delete_utxos_t = boost::unordered_flat_map<utxo_key_t, kth::domain::chain::input>;

std::tuple<to_insert_utxos_t, to_delete_utxos_t, size_t> process_in_block(std::vector<kth::domain::chain::transaction>& txs) {
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
        auto tx_hash = tx.hash();
        utxo_key_t key;
        // copy the transaction hash into the key
        std::copy(tx_hash.begin(), tx_hash.end(), key.begin());

        for (auto&& input : tx.inputs()) {
            auto const& prev_out = input.previous_output();
            auto const idx = prev_out.index();
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
        in_block_utxos
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
    size_t block_height = 0;

    utxo::utxo_db db;

    log_print("Opening DB ...\n");
    db.configure("utxo_interprocess_multiple", true); 
    log_print("DB opened ...\n");

    process(path,
        [&](auto const& tx_hashes, auto&& txs) {
            // log_print("txs.size() = {}\n", txs.size());
            // log_print("do something with txs\n");

            log_print("Processing block with {} transactions...\n", txs.size());
            auto const [
                to_insert, 
                to_delete,
                in_block_utxos_count
            ] = process_in_block(txs);

            log_print("Processed block with {} inputs and {} outputs\n", 
                      to_delete.size(), to_insert.size());

            log_print("deleting inputs...");
            // first, delete the inputs
            for (auto const& [k, v] : to_delete) {
                db.erase(k, block_height);
            }

            log_print("Inserting outputs...");
            // then, insert the outputs
            for (auto const& [k, v] : to_insert) {
                db.insert(k, v.to_data(), block_height);
            }

        },
        [&]() {
            // log_print("post processing\n");
        },
        total_inputs, total_outputs, partial_inputs, partial_outputs);

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
