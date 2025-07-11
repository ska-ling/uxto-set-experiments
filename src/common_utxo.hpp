#ifndef COMMON_UTXO_HPP
#define COMMON_UTXO_HPP

#include <array>
#include <cstdint>
#include <span>
#include <iostream>
#include <algorithm>
#include <fmt/format.h>

namespace utxo {

// Same key type as your custom implementation
using utxo_key_t = std::array<uint8_t, 36>;
using span_bytes = std::span<uint8_t const>;

inline
void print_key(utxo_key_t const& key) {
    // first 32 bytes are the transaction hash, print in hex reversed
    for (size_t i = 0; i < 32; ++i) {
        log_print("{:02x}", key[31 - i]);
    }   
    // the last 4 bytes are the output index, print as integer
    uint32_t output_index = 0;
    std::copy(key.end() - 4, key.end(), reinterpret_cast<uint8_t*>(&output_index));
    log_print(":{}", output_index);
    log_print("\n");
}

} // namespace utxo

#endif // COMMON_UTXO_HPP