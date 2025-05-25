#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <array>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "utxo_model");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, "/home/fernando/dev/utxo-experiments/model.onnx", session_options);

    // === Batch input: 3 UTXOs ===
    std::vector<std::vector<float>> batch_inputs = {
        {
            6.3010f, 148.0f, 13513.0f, 31.0f, 6.0f, 0.0f, 4.0f, 0.8f,
            1.0f, 0.0f, 0.0f, 0, 0, 1, 0, 0, 0  // small
        },
        {
            8.0000f, 100.0f, 500000.0f, 50.0f, 5.0f, 1.0f, 10.0f, 0.2f,
            0.0f, 1.0f, 8.0f, 0, 0, 0, 0, 1, 0  // large
        },
        {
            4.5000f, 80.0f, 100.0f, 12.0f, 7.0f, 0.0f, 1.0f, 0.6f,
            1.0f, 0.0f, 0.0f, 1, 0, 0, 0, 0, 0  // dust
        }
    };

    const size_t batch_size = batch_inputs.size();
    const size_t feature_count = batch_inputs[0].size();
    std::vector<float> input_flat;
    for (const auto& row : batch_inputs)
        input_flat.insert(input_flat.end(), row.begin(), row.end());

    std::array<int64_t, 2> input_shape{static_cast<int64_t>(batch_size), static_cast<int64_t>(feature_count)};
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_flat.data(),
        input_flat.size(),
        input_shape.data(),
        input_shape.size()
    );

    Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, allocator);
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1
    );

    float* results = output_tensors.front().GetTensorMutableData<float>();

    std::cout << "\n🔥 Predicted spend probabilities (batch):" << std::endl;
    for (size_t i = 0; i < batch_size; ++i)
        std::cout << "  UTXO " << i << ": " << results[i] << std::endl;

    return 0;
}
