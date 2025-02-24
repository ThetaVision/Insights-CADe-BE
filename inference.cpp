#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "onnxruntime_cxx_api.h"

int main() {
    // Initialize ONNX Runtime environment and session options.
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SegmentationModel");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Path to your ONNX segmentation model.
    const char* model_path = "C:\Users\20195435\Documents\theta\projects\cosmo\Insights-CADe-BE\experiments\baseline\endo_caformer_fpn.onnx";
    Ort::Session session(env, model_path, session_options);

    // Create a default allocator.
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input node name and type information.
    size_t num_input_nodes = session.GetInputCount();
    if (num_input_nodes == 0) {
        std::cerr << "Error: Model has no inputs." << std::endl;
        return -1;
    }
    char* input_name = session.GetInputName(0, allocator);
    Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
    auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_node_dims = tensor_info.GetShape();

    // If the model has dynamic dimensions (i.e. -1), set them to a fixed size (e.g., 1).
    for (auto &dim : input_node_dims) {
        if (dim < 0) {
            dim = 1;
        }
    }

    // Calculate total number of elements for the input tensor.
    size_t input_tensor_size = 1;
    for (const auto& dim : input_node_dims)
        input_tensor_size *= dim;

    // Prepare random input data.
    std::vector<float> input_tensor_values(input_tensor_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.f, 1.f);
    for (auto& value : input_tensor_values)
        value = dis(gen);

    // Retrieve all output node names.
    size_t num_output_nodes = session.GetOutputCount();
    std::vector<const char*> output_names(num_output_nodes);
    for (size_t i = 0; i < num_output_nodes; i++) {
        output_names[i] = session.GetOutputName(i, allocator);
    }

    // Run inference multiple times and time each run.
    const int num_runs = 10;
    double total_inference_time = 0.0;
    for (int run = 0; run < num_runs; ++run) {
        // Start timing.
        auto start = std::chrono::high_resolution_clock::now();

        // Create input tensor object from data values.
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_size,
            input_node_dims.data(),
            input_node_dims.size()
        );

        // Run inference. Note: session.Run requires arrays of input and output names.
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            &input_name,
            &input_tensor,
            1,
            output_names.data(),
            num_output_nodes
        );

        // End timing.
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        total_inference_time += elapsed.count();
        std::cout << "Run " << run + 1 << " took " << elapsed.count() << " ms." << std::endl;
    }

    // Compute and display the average inference time.
    double average_time = total_inference_time / num_runs;
    std::cout << "Average inference time: " << average_time << " ms." << std::endl;

    // Free allocated names.
    allocator.Free(input_name);
    for (auto name : output_names) {
        allocator.Free(const_cast<void*>(reinterpret_cast<const void*>(name)));
    }

    return 0;
}
