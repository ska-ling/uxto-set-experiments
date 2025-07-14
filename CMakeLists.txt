cmake_minimum_required(VERSION 3.15)
project(uxto-set-experiments CXX)
set(CMAKE_CXX_STANDARD 23)


find_package(node REQUIRED)
find_package(fmt REQUIRED)
find_package(absl REQUIRED)
find_package(Boost REQUIRED)
find_package(leveldb REQUIRED)
# find_package(robin-hood-hashing REQUIRED)
# find_package(robin_hood_hashing REQUIRED)
find_package(unordered_dense REQUIRED)
find_package(svector REQUIRED)

add_definitions(-DKTH_CURRENCY_BCH)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")


# set(BENCH_PROJECT_NAME bench_utxo_db)
set(BENCH_LOOKUP_PROJECT_NAME bench_lookup)
set(COMPARISON_PROJECT_NAME bench_db_comparison)
set(TEST_LEVELDB_NAME test_leveldb)
# set(MIGRATE_TO_LEVELDB migrate_to_leveldb)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/parlayhash)
# ---------------------------------------------------------------

# add_executable(${PROJECT_NAME} src/main.cpp)
# add_executable(${BENCH_PROJECT_NAME} src/bench_db.cpp)
add_executable(${BENCH_LOOKUP_PROJECT_NAME} src/bench_lookup.cpp)
# add_executable(${COMPARISON_PROJECT_NAME} src/bench_db_comparison.cpp)
# add_executable(${TEST_LEVELDB_NAME} src/test_leveldb.cpp)
# add_executable(${MIGRATE_TO_LEVELDB} src/migrate_to_leveldb.cpp)

# target_link_libraries(${PROJECT_NAME} PUBLIC node::node)
# target_link_libraries(${PROJECT_NAME} PUBLIC fmt::fmt-header-only)
# target_link_libraries(${PROJECT_NAME} PUBLIC leveldb::leveldb)
# # target_link_libraries(${PROJECT_NAME} PUBLIC ${JEMALLOC_LIB})


# target_link_libraries(${BENCH_PROJECT_NAME} PUBLIC node::node)
# target_link_libraries(${BENCH_PROJECT_NAME} PUBLIC fmt::fmt-header-only)
# target_link_libraries(${BENCH_PROJECT_NAME} PUBLIC leveldb::leveldb)
# # target_link_libraries(${BENCH_PROJECT_NAME} PUBLIC ${JEMALLOC_LIB})

target_link_libraries(${BENCH_LOOKUP_PROJECT_NAME} PUBLIC node::node)
target_link_libraries(${BENCH_LOOKUP_PROJECT_NAME} PUBLIC fmt::fmt-header-only)
target_link_libraries(${BENCH_LOOKUP_PROJECT_NAME} PUBLIC leveldb::leveldb)
# target_link_libraries(${BENCH_LOOKUP_PROJECT_NAME} PUBLIC ${JEMALLOC_LIB})

# target_link_libraries(${COMPARISON_PROJECT_NAME} PUBLIC node::node)
# target_link_libraries(${COMPARISON_PROJECT_NAME} PUBLIC fmt::fmt-header-only)
# target_link_libraries(${COMPARISON_PROJECT_NAME} PUBLIC leveldb::leveldb)
# # target_link_libraries(${COMPARISON_PROJECT_NAME} PUBLIC ${JEMALLOC_LIB})

# target_link_libraries(${TEST_LEVELDB_NAME} PUBLIC fmt::fmt-header-only)
# target_link_libraries(${TEST_LEVELDB_NAME} PUBLIC leveldb::leveldb)


# target_link_libraries(${MIGRATE_TO_LEVELDB} PUBLIC node::node)
# target_link_libraries(${MIGRATE_TO_LEVELDB} PUBLIC fmt::fmt-header-only)
# target_link_libraries(${MIGRATE_TO_LEVELDB} PUBLIC leveldb::leveldb)
# # target_link_libraries(${MIGRATE_TO_LEVELDB} PUBLIC ${JEMALLOC_LIB})


# install(TARGETS ${PROJECT_NAME} DESTINATION "."
#         RUNTIME DESTINATION bin
#         ARCHIVE DESTINATION lib
#         LIBRARY DESTINATION lib
#         )

# install(TARGETS ${BENCH_PROJECT_NAME} DESTINATION "."
#         RUNTIME DESTINATION bin
#         ARCHIVE DESTINATION lib
#         LIBRARY DESTINATION lib
# )

install(TARGETS ${BENCH_LOOKUP_PROJECT_NAME} DESTINATION "."
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
)

# install(TARGETS ${COMPARISON_PROJECT_NAME} DESTINATION "."
#         RUNTIME DESTINATION bin
#         ARCHIVE DESTINATION lib
#         LIBRARY DESTINATION lib
# )

# install(TARGETS ${TEST_LEVELDB_NAME} DESTINATION "."
#         RUNTIME DESTINATION bin
#         ARCHIVE DESTINATION lib
#         LIBRARY DESTINATION lib
# )

# install(TARGETS ${MIGRATE_TO_LEVELDB} DESTINATION "."
#         RUNTIME DESTINATION bin
#         ARCHIVE DESTINATION lib
#         LIBRARY DESTINATION lib
# )
