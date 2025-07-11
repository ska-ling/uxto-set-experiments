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


#include "leveldb_v1.hpp"
#include "interprocess_multiple_v12.hpp"
#include <iostream>


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
    init_log_file("migrate_to_leveldb");
    log_print("=== Migrating UTXO database to LevelDB ===\n");
    


    // Inicializar la base custom en modo solo lectura
    utxo::utxo_db db_custom;
    db_custom.configure("utxo_interprocess_multiple", false);

    // Inicializar la base LevelDB en modo escritura (limpia el directorio)
    utxo::utxo_db_leveldb db_leveldb;
    db_leveldb.configure("./leveldb_utxos", true);

    size_t count = 0;
    // func(key, value.get_data(), value.block_height, Index, current_versions_[Index]);
    db_custom.for_each_entry([&](utxo::utxo_key_t const& key, auto const& data, uint32_t height, size_t container, size_t version) {
        // Insertar en LevelDB
        bool ok = db_leveldb.insert(key, std::span<const uint8_t>(data.data(), data.size()), height);
        if (!ok) {
            std::cerr << "Error al insertar clave en LevelDB" << std::endl;
        }
        ++count;
        if (count % 100000 == 0) {
            std::cout << count << " elementos migrados..." << std::endl;
        }
    });
    std::cout << "Migración completa. Total: " << count << " elementos." << std::endl;
    return 0;
}
