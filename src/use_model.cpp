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
        6.3010f,   // log_value
        148.0f,    // total_script_size
        13513.0f,  // script_efficiency
        31.0f,     // block_time_proxy
        6.0f,      // creation_epoch
        0.0f,      // is_coinbase
        4.0f,      // block_density
        0.8f,      // value_percentile_in_block
        1.0f,      // is_likely_change
        0.0f,      // is_likely_savings
        0.0f,      // coinbase_maturity_factor

        // One-hot: value_class_dust, micro, small, medium, large, whale
        0.0f,      // value_class_dust
        0.0f,      // value_class_micro
        1.0f,      // value_class_small
        0.0f,      // value_class_medium
        0.0f,      // value_class_large
        0.0f       // value_class_whale
    };

    std::array<int64_t, 2> input_shape{1, static_cast<int64_t>(input_features.size())};
    Ort::AllocatorWithDefaultOptions allocator;

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_features.data(),
        input_features.size(),
        input_shape.data(),
        input_shape.size()
    );


    // const char* input_names[] = {"X"};
    Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
    const char* input_names[] = {input_name.get()};

    // const char* output_names[] = {session.GetOutputName(0, allocator)};
    Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, allocator);
    const char* output_names[] = {output_name.get()};


    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1
    );

    float* result = output_tensors.front().GetTensorMutableData<float>();
    std::cout << "🔥 Predicted spend probability: " << result[0] << std::endl;

    return 0;
}
