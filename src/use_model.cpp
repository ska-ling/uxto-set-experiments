#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <array>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "utxo_model");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Ruta al modelo ONNX exportado desde Python
    Ort::Session session(env, "/home/fernando/dev/utxo-experiments/model.onnx", session_options);

    // === Input features ===
    // Orden exacto según classifier.feature_columns (float32)
    std::vector<float> input_features = {
        6.3010,   // log_value
        148.0,    // total_script_size
        13513.0,  // script_efficiency
        31.0,     // block_time_proxy
        0.0,      // is_coinbase
        4.0,      // block_density
        0.8,      // value_percentile_in_block
        1.0,      // is_likely_change
        0.0,      // is_likely_savings
        6.3010,   // coinbase_maturity_factor

        // one-hot value_class_dust, micro, small, medium, large, big, whale
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,

        // creation_epoch
        5.0
    };

    std::array<int64_t, 2> input_shape{1, static_cast<int64_t>(input_features.size())};
    Ort::AllocatorWithDefaultOptions allocator;

    // Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    //     allocator, input_features.data(), input_features.size(),
    //     input_shape.data(), input_shape.size()
    // );

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_features.data(),
        input_features.size(),
        input_shape.data(),
        input_shape.size()
    );


    const char* input_names[] = {"input"};
    const char* output_names[] = {session.GetOutputName(0, allocator)};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1
    );

    float* result = output_tensors.front().GetTensorMutableData<float>();
    std::cout << "🔥 Predicted spend probability: " << result[0] << std::endl;

    return 0;
}
