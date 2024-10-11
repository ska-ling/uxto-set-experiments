#ifndef UTXO_SET_COMMON_HPP_
#define UTXO_SET_COMMON_HPP_

#include <filesystem>
#include <fstream>
#include <random>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/std.h>

#include <kth/domain/chain/block.hpp>
#include <kth/domain/chain/input_point.hpp>
#include <kth/consensus.hpp>

#ifdef JEMALLOC
#include <jemalloc/jemalloc.h>

inline
void print_memory_usage(std::string_view message) {
    size_t epoch = 1;
    size_t sz = sizeof(size_t);
    size_t allocated = 0;
    size_t active = 0;
    size_t resident = 0;
    size_t retained = 0;

    mallctl("thread.tcache.flush", NULL, NULL, NULL, 0);
    mallctl("epoch", NULL, NULL, &epoch, sizeof(epoch));
    mallctl("stats.allocated", &allocated, &sz, nullptr, 0);
    mallctl("stats.active", &active, &sz, nullptr, 0);
    mallctl("stats.resident", &resident, &sz, nullptr, 0);
    mallctl("stats.retained", &retained, &sz, nullptr, 0);

    double allocated_mb = allocated / 1'000'000.0;
    double active_mb = active / 1'000'000.0;
    double resident_mb = resident / 1'000'000.0;
    double retained_mb = retained / 1'000'000.0;

    fmt::print("{:<60} - allocated: {:>10} - active: {:>10} - resident: {:>10} - retained: {:>10}\n", message, allocated_mb, active_mb, resident_mb, retained_mb);
}

inline
size_t get_allocated_memory() {
    size_t epoch = 1;
    size_t sz = sizeof(size_t);
    size_t allocated = 0;

    mallctl("thread.tcache.flush", NULL, NULL, NULL, 0);
    mallctl("epoch", NULL, NULL, &epoch, sizeof(epoch));
    mallctl("stats.allocated", &allocated, &sz, nullptr, 0);

    return allocated;
}

#else
inline
void print_memory_usage(std::string_view message) {
    fmt::print("{} - memory usage not available\n", message);
}
inline
size_t get_allocated_memory() {
    return 0;
}
#endif



using bytes_t = std::vector<uint8_t>;

uint8_t char2int(char input) {
    if(input >= '0' && input <= '9') return input - '0';
    if(input >= 'A' && input <= 'F') return input - 'A' + 10;
    if(input >= 'a' && input <= 'f') return input - 'a' + 10;
    throw std::invalid_argument("Invalid input string");
}

void hex2bin(char const* src, uint8_t* target) {
    while(*src && src[1]) {
        *(target++) = char2int(*src)*16 + char2int(src[1]);
        src += 2;
    }
}

bytes_t hex2vec(char const* src, size_t n) {
    bytes_t bytes(n / 2);
    hex2bin(src, bytes.data());
    return bytes;
}

std::string bytes2hex(bytes_t const& bytes) {
    std::string hex(bytes.size() * 2, '\0');
    for (size_t i = 0; i < bytes.size(); ++i) {
        hex[i * 2] = "0123456789ABCDEF"[bytes[i] >> 4];
        hex[i * 2 + 1] = "0123456789ABCDEF"[bytes[i] & 0xF];
    }
    return hex;
}

std::vector<bytes_t> get_blocks_raw_from_n(std::filesystem::path blocks_filename, size_t from_block, size_t n) {
    std::vector<bytes_t> blocks_raw;
    blocks_raw.reserve(n);

    std::ifstream file(std::string{blocks_filename});
    if ( ! file) {
        throw std::runtime_error("Unable to open file: " + blocks_filename.string());
    }

    std::string str;
    size_t i = 0;
    while (i < from_block && std::getline(file, str)) {
        ++i;
    }

    while (std::getline(file, str)) {
        auto bytes = hex2vec(str.data(), str.size());
        blocks_raw.push_back(std::move(bytes));
        if (blocks_raw.size() == n) {
            break;
        }
    }

    return blocks_raw;
}


using TransactionReadResult = std::tuple<
    std::vector<kth::domain::chain::transaction>,
    size_t,
    size_t,
    size_t,
    size_t>;

void report_block_memory_usage(kth::domain::chain::block const& blk, size_t global_block_index, size_t before_parse_block, size_t after_parse_block, size_t block_raw_size) {

#ifdef MEMORY_USAGE_CHECK
    size_t inputs = 0;
    size_t outputs = 0;
    for (auto const& tx : blk.transactions()) {
        inputs += tx.inputs().size();
        outputs += tx.outputs().size();
    }

    size_t const block_memory = after_parse_block - before_parse_block;
    // if (block_memory > block_raw_size * 1.1) {
    if (block_memory > block_raw_size * 10) {
        fmt::print("ALERT: Memory allocation for block {} is too high: {} bytes.\n", global_block_index, block_memory);
        fmt::print("before_parse_block: {} bytes.\n", before_parse_block);
        fmt::print("after_parse_block: {} bytes.\n", after_parse_block);
        fmt::print("block_raw_size: {} bytes.\n", block_raw_size);
        fmt::print("transactions:     {}\n", blk.transactions().size());
        fmt::print("inputs:           {}\n", inputs);
        fmt::print("outputs:          {}\n", outputs);
    }
#endif
}

TransactionReadResult get_n_transactions(std::filesystem::path const& path, size_t block_from, size_t tx_from, size_t n) {
    constexpr size_t file_step = 10'000;    //TODO: hardcoded values
    constexpr size_t file_max = 489'999;

    size_t before_tx_reserve = get_allocated_memory();

    std::vector<kth::domain::chain::transaction> transactions;
    transactions.reserve(n);

    size_t after_tx_reserve = get_allocated_memory();

    auto const remaining = [&](){ return n - transactions.size(); };

    size_t file_index = block_from / file_step;
    size_t current_file_start = file_index * file_step;
    size_t current_block_index = block_from % file_step;
    size_t global_block_index = block_from;
    size_t total_raw_block_bytes = 0;
    size_t total_parsed_block_bytes = 0;

    while (true) {
        if (current_file_start > file_max) {
            break;
        }
        if (transactions.size() >= n) {
            break;
        }
        size_t const current_file_end = std::min(current_file_start + file_step - 1, file_max);
        std::filesystem::path const blocks_file = path / fmt::format("block-raw-{}-{}.csv", current_file_start, current_file_end);
        fmt::print("Processing file {}\n", blocks_file);

        size_t const blocks_to_read = std::min(remaining(), file_step);
        auto blocks_raw = get_blocks_raw_from_n(blocks_file, current_block_index, blocks_to_read);
        // print_memory_usage("get_n_transactions() after calling get_blocks_raw_from_n()");

        for (size_t i = 0; i < blocks_raw.size(); ++i) {
            auto& block_raw = blocks_raw[i];
            total_raw_block_bytes += block_raw.size();
            size_t const before_parse_block = get_allocated_memory();
            kth::domain::chain::block blk;

            kth::domain::entity_from_data(blk, block_raw);
            auto const valid = blk.is_valid();

            if ( ! valid) {
                // Should not happen
                fmt::print("****** INVALID BLOCK ******\n");
                std::terminate();
            }

            size_t const after_parse_block = get_allocated_memory();
            size_t const block_memory = after_parse_block - before_parse_block;

            // report_block_memory_usage(blk, global_block_index, before_parse_block, after_parse_block, block_raw.size());

            auto& txs = blk.transactions();
            size_t const start_index = global_block_index == block_from ? tx_from : 0;
            size_t const end_index = std::min(txs.size(), start_index + remaining());
            if (start_index == 0) {
                // This is the first time block (global_block_index) is processed
                total_parsed_block_bytes += block_memory;
            }

            if (start_index >= txs.size()) {
                // Skip this block if the start index is out of range
                // This should not happen
                fmt::print("******** Start index out of range ********\n");
                throw std::runtime_error("Start index out of range");
                continue;
            }
            size_t old_transaction_size = transactions.size();
            size_t old_txs_size = txs.size();

            std::move(txs.begin() + start_index, txs.begin() + end_index, std::back_inserter(transactions));

            if (transactions.size() >= n) {
                size_t const next_tx_index = end_index == txs.size() ? 0 : end_index;
                size_t const next_block_index = global_block_index + (end_index == txs.size() ? 1 : 0);

                blk.reset();
                fmt::print("(1) Returning transactions_collected: {} - block {} - tx {}\n", transactions.size(), next_block_index, next_tx_index);
                return {std::move(transactions), next_block_index, next_tx_index, total_raw_block_bytes, total_parsed_block_bytes};
            }

            // print_memory_usage("get_n_transactions() before resetting block");
            blk.reset();
            // print_memory_usage("get_n_transactions() after resetting block");
            ++global_block_index;
        }

        current_file_start += file_step;
        current_block_index = 0;
    }

    fmt::print("Total transactions collected: {}\n", transactions.size());
    fmt::print("(2) Returning transactions_collected: {} - file: {} - block {} - tx {}\n", transactions.size(), current_file_start, current_block_index, 0);

    size_t reservation_mb = (after_tx_reserve - before_tx_reserve) / 1'000'000;
    fmt::print("Memory reservation for {} transactions: {} MB\n", transactions.size(), reservation_mb);

    size_t const current_allocated = get_allocated_memory();
    size_t const allocated_mb = current_allocated / 1'000'000;
    fmt::print("Current memory usage: {} MB\n", allocated_mb);
    fmt::print("Total raw block bytes: {} MB\n", total_raw_block_bytes / 1'000'000);
    fmt::print("Total parsed block bytes: {} MB\n", total_parsed_block_bytes / 1'000'000);

    return {std::move(transactions), block_from, 0, total_raw_block_bytes, total_parsed_block_bytes};
}

template <typename ProcessTxs, typename PostProcessing>
void process(std::filesystem::path const& path, ProcessTxs process_txs, PostProcessing post_processing, size_t& total_inputs, size_t& total_outputs, size_t& partial_inputs, size_t& partial_outputs) {
    // constexpr size_t file_max = 780'000;
    // constexpr size_t file_max = 99'000;          //TODO: hardcoded values
    // constexpr size_t file_step = 10000;
    constexpr size_t max_blocks = 1000;

    std::random_device rd;
    std::mt19937 gen(rd());
    // std::uniform_int_distribution<int> dis(500'000, 1'000'000); //TODO: hardcoded values
    std::uniform_int_distribution<int> dis(50'000, 100'000); //TODO: hardcoded values

    size_t block_from = 0;
    size_t tx_from = 0;
    size_t input_count = 0;
    size_t output_count = 0;
    size_t total_transactions = 0;
    double total_time = 0.0;
    size_t total_raw_block_bytes = 0;
    size_t total_parsed_block_bytes = 0;

    while (true) {
        size_t const transaction_count = dis(gen);
        fmt::print("Reading {} real transactions from files ...\n", transaction_count);
        auto [transactions,
              tmp_block_from,
              tmp_tx_from,
              partial_raw_block_bytes,
              partial_parsed_block_bytes] = get_n_transactions(path, block_from, tx_from, transaction_count);
        fmt::print("Processing {} transactions ...\n", transactions.size());

        if (transactions.empty()) {
            fmt::print("No more transactions to read\n");
            break;
        }

        auto partial_transactions = transactions.size();
        auto partial_inputs = 0;
        auto partial_outputs = 0;
        total_transactions += transactions.size();
        for (auto const& tx : transactions) {
            partial_inputs += tx.inputs().size();
            partial_outputs += tx.outputs().size();
        }
        input_count += partial_inputs;
        output_count += partial_outputs;

        // Calculating Transaction Hashes
        std::vector<kth::hash_digest> tx_hashes;
        tx_hashes.reserve(transactions.size());
        for (auto const& tx : transactions) {
            tx_hashes.push_back(tx.hash());
        }

        auto const start = std::chrono::high_resolution_clock::now();
        process_txs(tx_hashes, transactions);
        auto const end = std::chrono::high_resolution_clock::now();
        double const partial_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        total_time += partial_time;

        total_raw_block_bytes += partial_raw_block_bytes;
        total_parsed_block_bytes += partial_parsed_block_bytes;


        post_processing();

        fmt::print("Partial Transactions: {:7}\n", partial_transactions);
        fmt::print("Partial Inputs:       {:7}\n", partial_inputs);
        fmt::print("Partial Outputs:      {:7}\n", partial_outputs);
        fmt::print("Total Transactions:   {:7}\n", total_transactions);
        fmt::print("Total Inputs:         {:7}\n", input_count);
        fmt::print("Total Outputs:        {:7}\n", output_count);
        fmt::print("Partial Time:         {:7}ns - {:7}ms - {:7}s\n", partial_time, partial_time / 1'000'000, partial_time / 1'000'000'000);
        fmt::print("Total Time:           {:7}ns - {:7}ms - {:7}s\n", total_time, total_time / 1'000'000, total_time / 1'000'000'000);
        fmt::print("Partial TXs/sec:      {:010}\n", double(partial_transactions) * 1'000'000'000.0 / partial_time);
        fmt::print("Partial Inputs/sec:   {:010}\n", double(partial_inputs) * 1'000'000'000.0 / partial_time);
        fmt::print("Partial Outputs/sec:  {:010}\n", double(partial_outputs) * 1'000'000'000.0 / partial_time);
        fmt::print("Total TXs/sec:        {:010}\n", double(total_transactions) * 1'000'000'000.0 / total_time);
        fmt::print("Total Inputs/sec:     {:010}\n", double(input_count) * 1'000'000'000.0 / total_time);
        fmt::print("Total Outputs/sec:    {:010}\n", double(output_count) * 1'000'000'000.0 / total_time);

        fmt::print("Partial Raw Block Bytes: {} MB\n", partial_raw_block_bytes / 1'000'000);
        fmt::print("Partial Parsed Block Bytes: {} MB\n", partial_parsed_block_bytes / 1'000'000);
        fmt::print("Ratio: {}\n", double(partial_parsed_block_bytes) / partial_raw_block_bytes);
        fmt::print("Total Raw Block Bytes: {} MB\n", total_raw_block_bytes / 1'000'000);
        fmt::print("Total Parsed Block Bytes: {} MB\n", total_parsed_block_bytes / 1'000'000);
        fmt::print("Ratio: {}\n", double(total_parsed_block_bytes) / total_raw_block_bytes);

        block_from = tmp_block_from;
        tx_from = tmp_tx_from;
    }
    fmt::print("Total transactions: {}\n", total_transactions);
    fmt::print("Total inputs:       {}\n", input_count);
    fmt::print("Total outputs:      {}\n", output_count);
}

#endif // UTXO_SET_COMMON_HPP_
