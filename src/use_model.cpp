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

    // Obtener nombres dinámicamente
    Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name_0 = session.GetOutputNameAllocated(0, allocator); // label
    Ort::AllocatedStringPtr output_name_1 = session.GetOutputNameAllocated(1, allocator); // probabilities

    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name_0.get(), output_name_1.get()};

    // Ejecutar sesión
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 2
    );

    // Obtener probabilidades
    float* probabilities = output_tensors[1].GetTensorMutableData<float>();

    std::cout << "🔥 Predicted spend probability (cold): " << probabilities[0] << std::endl;
    std::cout << "🔥 Predicted spend probability (hot) : " << probabilities[1] << std::endl;

    return 0;
}
