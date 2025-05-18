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
    size_t creation_block;
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
    output_file << "txid,index,creation_block,spent_block\n";
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

// Function to process a block
void process_block(std::string const& block_hex, size_t block_height) {
    auto block_bytes = hex2vec(block_hex.data(), block_hex.size());
    kth::byte_reader reader(block_bytes);
    auto blk_exp = kth::domain::chain::block::from_data(reader);
    auto const& blk = blk_exp.value();
    for (const auto& tx : blk.transactions()) {
        for (size_t i = 0; i < tx.outputs().size(); ++i) {
            auto key = create_utxo_key(tx.hash(), i);
            utxo_set[key] = {block_height};
        }

        for (const auto& input : tx.inputs()) {
            auto key = create_utxo_key(input.previous_output().hash(), input.previous_output().index());
            auto it = utxo_set.find(key);
            if (it != utxo_set.end()) {
                output_buffer.push_back(
                    fmt::format("{},{},{}\n", 
                        kth::encode_base16(key), 
                        it->second.creation_block, 
                        block_height
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
            process_block(line, current_file_start);
            ++block_height;
        }
    }

    // Write remaining buffer
    write_output_buffer();

    // Write remaining unspent UTXOs
    for (const auto& [key, utxo] : utxo_set) {
        output_buffer.push_back(fmt::format("{},{},Unspent\n", kth::encode_base16(key), utxo.creation_block));
    }

    write_output_buffer();

    fmt::print("Completed processing.\n");
    return 0;
}
