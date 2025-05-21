#include <fmt/core.h>

#include <utxo/common.hpp>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <array>
#include <string>
#include <filesystem>
#include <vector>
#include <fmt/core.h>
#include <fmt/format.h>
#include <kth/domain.hpp> // Knuth domain for transaction parsing

using UTXOKey = std::array<uint8_t, 36>;

struct UTXOEntry {
    uint32_t creation_block;
    uint64_t value;                  // Monto del output (satoshis)
    uint16_t locking_script_size;    // Tamaño del locking script (bytes)
    bool tx_coinbase;
    bool op_return;
};

// UTXO Map
std::unordered_map<UTXOKey, UTXOEntry> utxo_set;
std::vector<std::string> output_buffer;
constexpr size_t MAX_OUTPUT_BUFFER = 1'000'000; // 1 million rows
constexpr size_t MAX_OUTPUT_FILE_SIZE = 100'000'000; // 100 MB

std::string output_directory = "/home/fernando/dev/utxo-experiments/output";
std::string input_directory = "/home/fernando/dev/utxo-experiments/src";

// Function to create a UTXO Key
UTXOKey create_utxo_key(kth::hash_digest const& txid, uint32_t index) {
    UTXOKey key{};
    std::copy(txid.begin(), txid.end(), key.begin());
    auto index_ptr = reinterpret_cast<uint32_t*>(&key[32]);
    *index_ptr = index;
    return key;
}

size_t output_file_index = 0;
std::ofstream output_file;

void open_new_output_file() {
    fmt::print("Opening new output file...\n");
    if (output_file.is_open()) {
        output_file.close();
    }
    std::string filename = fmt::format("{}/utxo-history-{}.csv", output_directory, output_file_index++);
    output_file.open(filename);
    output_file << "creation_block,spent_block,value,locking_script_size,unlocking_script_size;tx_coinbase;op_return\n";
}

void write_output_buffer() {
    if (!output_file.is_open()) {
        open_new_output_file();
    }

    fmt::print("Writing {} entries to output file...\n", output_buffer.size());
    for (const auto& line : output_buffer) {
        output_file << line;
        if (output_file.tellp() >= MAX_OUTPUT_FILE_SIZE) {
            open_new_output_file();
        }
    }

    output_buffer.clear();
    fmt::print("Flushing output file...\n");
    output_file.flush();
    fmt::print("Output file flushed.\n");
}

bool is_op_return(kth::domain::chain::script const& script) {
    auto const& bytes = tx.outputs()[i].script().bytes();
    if (bytes.empty()) {
        return false;
    }
    if (bytes[0] == 0x6a) { // OP_RETURN
        fmt::print("OP_RETURN detected ***************************\n");
        print_hex(bytes);
        return true;
    }
    return false;
}

// Function to process a block
void process_block(std::string const& block_hex, uint32_t block_height) {
    auto block_bytes = hex2vec(block_hex.data(), block_hex.size());
    kth::byte_reader reader(block_bytes);
    auto blk_exp = kth::domain::chain::block::from_data(reader);
    auto const& blk = blk_exp.value();

    for (const auto& tx : blk.transactions()) {
        bool const tx_coinbase = tx.is_coinbase();
        for (size_t i = 0; i < tx.outputs().size(); ++i) {
            auto key = create_utxo_key(tx.hash(), i);
            uint64_t value = tx.outputs()[i].value();       // Monto del output (satoshis)
            uint16_t locking_script_size = tx.outputs()[i].script().serialized_size(false); // Tamaño del locking script

            // auto const& bytes = tx.outputs()[i].script().bytes();
            // fmt::print("Output script bytes: ");
            // print_hex(bytes);

            // auto output_script_pattern = uint8_t(tx.outputs()[i].script().output_pattern());
            // auto input_script_pattern = uint8_t(tx.outputs()[i].script().input_pattern());
            // // std::cout << "Output script pattern: " << (int)output_script_pattern << std::endl;
            // // std::cout << "Input script pattern: " << (int)input_script_pattern << std::endl;
            
            bool const is_op_return = is_op_return(tx.outputs()[i].script());
            
            utxo_set[key] = {block_height, value, locking_script_size, tx_coinbase, is_op_return};
        }

        for (const auto& input : tx.inputs()) {
            auto key = create_utxo_key(input.previous_output().hash(), input.previous_output().index());
            auto it = utxo_set.find(key);
            if (it != utxo_set.end()) {
                uint32_t unlocking_script_size = input.script().serialized_size(false); // Tamaño del unlocking script
                output_buffer.push_back(
                    fmt::format("{},{},{},{},{},{},{}\n",
                        it->second.creation_block, 
                        block_height, 
                        it->second.value, 
                        it->second.locking_script_size,
                        unlocking_script_size,
                        it->second.tx_coinbase,
                        it->second.op_return
                    )
                );
                utxo_set.erase(it);

                if (output_buffer.size() >= MAX_OUTPUT_BUFFER) {
                    fmt::print("Buffer size reached, writing to file...\n");
                    write_output_buffer();
                }
            }
        }
    }
}

// Main function
int main() {
    constexpr size_t file_step = 10'000;
    constexpr size_t file_max = 780'000;

    for (size_t current_file_start = 0; current_file_start <= file_max; current_file_start += file_step) {
        size_t current_file_end = std::min(current_file_start + file_step - 1, file_max);
        std::filesystem::path blocks_file = fmt::format("{}/block-raw-{}-{}.csv", input_directory, current_file_start, current_file_end);

        fmt::print("Processing file: {}\n", blocks_file);

        std::ifstream file(blocks_file);
        std::string line;

        uint32_t block_height = current_file_start;
        while (std::getline(file, line)) {
            process_block(line, block_height);
            ++block_height;
        }
    }

    // Write remaining buffer
    write_output_buffer();

    // Write remaining unspent UTXOs
    for (const auto& [key, utxo] : utxo_set) {
        output_buffer.push_back(fmt::format("{},Unspent,{},{},-{},{}\n", 
            utxo.creation_block, 
            utxo.value, 
            utxo.locking_script_size,
            utxo.tx_coinbase,
            utxo.op_return
        ));
    }


    write_output_buffer();

    fmt::print("Completed processing.\n");
    return 0;
}
