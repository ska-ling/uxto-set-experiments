#include <iostream>
#include <cassert>
#include "leveldb_v1.hpp"
#include "log.hpp"

std::ofstream log_file; // Define the global log file

int main() {
    // Open log file for testing
    log_file.open("leveldb_test.log");
    
    log_print("Starting LevelDB UTXO implementation test...\n");
    
    try {
        utxo::utxo_db_leveldb db;
        
        // Test 1: Configure and open database
        log_print("Test 1: Configure database...\n");
        bool success = db.configure("test_leveldb", true);
        assert(success && "Failed to configure database");
        log_print("✓ Database configured successfully\n");
        
        // Test 2: Insert some test data
        log_print("Test 2: Insert test data...\n");
        utxo::utxo_key_t test_key = {};
        std::string test_data = "Hello, LevelDB!";
        test_key[0] = 0x01; // Simple test key
        
        std::vector<uint8_t> test_value(test_data.begin(), test_data.end());
        success = db.insert(test_key, test_value, 12345);
        assert(success && "Failed to insert test data");
        log_print("✓ Test data inserted successfully\n");
        
        // Test 3: Find the inserted data
        log_print("Test 3: Find inserted data...\n");
        auto found_value = db.find(test_key, 12345);
        assert(found_value.has_value() && "Failed to find inserted data");
        
        std::string found_string(found_value->begin(), found_value->end());
        assert(found_string == test_data && "Found data doesn't match inserted data");
        log_print("✓ Data found and matches: {}\n", found_string);
        
        // Test 4: Delete the data
        log_print("Test 4: Delete data...\n");
        size_t deleted_count = db.erase(test_key, 12346);
        assert(deleted_count == 1 && "Failed to delete data");
        log_print("✓ Data deleted successfully\n");
        
        // Test 5: Verify data is gone
        log_print("Test 5: Verify data is deleted...\n");
        auto not_found = db.find(test_key, 12347);
        assert(!not_found.has_value() && "Data should not be found after deletion");
        log_print("✓ Data correctly not found after deletion\n");
        
        // Test 6: Test statistics
        log_print("Test 6: Test statistics...\n");
        auto stats = db.get_statistics();
        assert(stats.total_inserts == 1 && "Insert count should be 1");
        assert(stats.total_deletes == 1 && "Delete count should be 1");
        assert(stats.successful_finds == 1 && "Successful find count should be 1");
        assert(stats.failed_finds == 1 && "Failed find count should be 1");
        log_print("✓ Statistics are correct\n");
        
        // Print final statistics
        log_print("\nFinal statistics:\n");
        db.print_statistics();
        
        // Test 7: Close database
        log_print("Test 7: Close database...\n");
        db.close();
        log_print("✓ Database closed successfully\n");
        
        log_print("\nAll tests passed! ✅\n");
        
    } catch (std::exception const& e) {
        log_print("Test failed with exception: {}\n", e.what());
        log_file.close();
        return 1;
    }
    
    log_file.close();
    return 0;
}
