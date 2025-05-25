#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <array>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "utxo_model");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    Ort::Session session(env, "/home/fernando/dev/utxo-experiments/model.onnx", session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // === Lista de entradas (batch size = 3 por ejemplo)
    std::vector<std::vector<float>> batch_inputs = {
        {
            6.3010f, 148.0f, 13513.0f, 31.0f, 6.0f, 0.0f, 4.0f, 0.8f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f
        },
        {
            5.8451f, 98.0f, 10200.0f, 88.0f, 5.0f, 1.0f, 3.0f, 0.2f, 0.0f, 1.0f, 1.2f,
            0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f
        },
        {
            7.0000f, 80.0f, 200000.0f, 10.0f, 2.0f, 0.0f, 1.0f, 0.9f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f
        }
    };

    const size_t batch_size = batch_inputs.size();
    const size_t feature_dim = batch_inputs[0].size();

    // Aplanar a vector 1D
    std::vector<float> input_tensor_values;
    for (const auto& row : batch_inputs) {
        input_tensor_values.insert(input_tensor_values.end(), row.begin(), row.end());
    }

    std::array<int64_t, 2> input_shape{static_cast<int64_t>(batch_size), static_cast<int64_t>(feature_dim)};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    // Nombres
    Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name_0 = session.GetOutputNameAllocated(0, allocator); // label
    Ort::AllocatedStringPtr output_name_1 = session.GetOutputNameAllocated(1, allocator); // probabilities

    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name_0.get(), output_name_1.get()};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 2
    );

    // === Interpretar probabilidades (2 floats por entrada)
    float* probabilities = output_tensors[1].GetTensorMutableData<float>();

    std::cout << "\n🔥 Predicted spend probabilities (batch):\n";
    for (size_t i = 0; i < batch_size; ++i) {
        float cold = probabilities[i * 2 + 0];
        float hot  = probabilities[i * 2 + 1];
        std::cout << "  UTXO " << i << " -> Cold: " << cold << " | Hot: " << hot << std::endl;
    }

    return 0;
}
