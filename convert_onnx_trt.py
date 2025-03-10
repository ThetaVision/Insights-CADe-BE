import argparse
import torch
import onnx
import onnxruntime
import tensorrt as trt
import numpy as np
from models import MetaFormer
import os
import time
from utils import common


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def quantize_model(model, args):
    # Quantize the model
    q_dtype = None
    if args.precision == "int8":
        q_dtype = torch.qint8

        # qconfig_spec = None
        supported_modules = {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}

        model = torch.ao.quantization.quantize_dynamic(
                            model,  # the original model
                            supported_modules,
                            dtype=q_dtype,
        )

    elif args.precision == "fp16":
        q_dtype = torch.float16
        # Convert entire model to FP16
        model = model.half()
        
        # Revert problematic layers back to FP32.
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
                module.float()
    else:
        print("No quantization needed.")
          
    return model


def export_to_onnx(torch_model, args):
    torch_model = torch_model.float()
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # Set precision
    if args.precision == "fp16":
        torch_model = torch_model.half()  # Convert model to FP16
        x = torch.randn(args.batch_size, 3, 256, 256, requires_grad=True).half()  # FP16 input
        print("Exporting model in FP16 precision.")
    elif args.precision == "int8":
        torch_model = torch.quantization.quantize_dynamic(
            torch_model, {torch.nn.Linear}, dtype=torch.qint8
        )  # Convert model to INT8 dynamically
        x = torch.randn(args.batch_size, 3, 256, 256, requires_grad=True)  # INT8 inputs still need FP32
        print("Exporting model in INT8 precision (dynamic quantization).")
    else:  # Default: FP32
        x = torch.randn(args.batch_size, 3, 256, 256, requires_grad=True)
        print("Exporting model in FP32 precision.")

    # Get model results
    torch_out_cls, torch_out_seg = torch_model(x)

    # Quantize the model
    quantized_model = quantize_model(torch_model, args)

    # confirm model is quantized

    # Export the quantized model to ONNX
    torch.onnx.export(quantized_model,           # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  args.onnx_model,           # where to save the model
                  export_params=True,        # store trained weights inside the model
                  opset_version=11,          # ONNX version
                  do_constant_folding=True,  # Optimize constant expressions
                  input_names=['input'],     # Input names
                  output_names=['output'],   # Output names
                  dynamic_axes={'input': {0: 'batch_size'},    # Variable batch size
                                'output': {0: 'batch_size'}})

    # Load and check ONNX model
    onnx_model = onnx.load(args.onnx_model)
    onnx.checker.check_model(onnx_model, full_check=True)
    print('ONNX model is well-formed.')

    # ONNX Runtime Inference
    ort_session = onnxruntime.InferenceSession(args.onnx_model)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    ort_outs_cls, ort_outs_seg = ort_outs
    if args.precision == "int8" or args.precision == "fp16":
        np.testing.assert_allclose(to_numpy(torch_out_cls), ort_outs_cls, rtol=4e-02, atol=1e-05)
        np.testing.assert_allclose(to_numpy(torch_out_seg), ort_outs_seg, rtol=4e-02, atol=1e-05)
        print('The outputs are similar enough')
    else: 
        np.testing.assert_allclose(to_numpy(torch_out_cls), ort_outs_cls, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(to_numpy(torch_out_seg), ort_outs_seg, rtol=1e-03, atol=1e-05)

        print('The outputs are similar!')

    return onnx_model

def export_to_trt(args):
    # # convert onnx model to tensorrt model
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # create builder
    builder = trt.Builder(TRT_LOGGER)
    # builder.max_workspace_size = 1 << 30
    # builder.max_batch_size = args.batch_size
    # builder.fp32_mode = True

    # Explicit batch flag
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    # Create network with EXPLICIT_BATCH
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # parser.parse(onnx_model.SerializeToString())

    success = parser.parse_from_file(args.onnx_model)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        raise Exception('Failed to parse the ONNX file')

    # build engine
    config = builder.create_builder_config()
    # engine = builder.build_cuda_engine(network)

    # Set precision mode
    if args.precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        print("Using FP16 precision.")
    elif args.precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        print("Using INT8 precision (calibration required).")
    else:
        print("Using FP32 precision (default).")

    # Define an optimization profile for dynamic input shapes.
    # Replace "input_tensor_name" with the actual name of your input.
    profile = builder.create_optimization_profile()

    # Set the minimum, optimum, and maximum shapes.
    # For example, if your model's input shape is (batch, channels, height, width):
    profile.set_shape("input", 
                    min=(1, 3, 256, 256),   # Minimum dimensions
                    opt=(1, 3, 256, 256),   # Optimal dimensions
                    max=(8, 3, 256, 256))   # Maximum dimensions

    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build serialized engine.")

    with open(args.trt_model, "wb") as f:
        f.write(serialized_engine)

    print('TensorRT model is saved!')


def inference_speed_test_cpu(torch_model, args):

    # Assume to_numpy, common.allocate_buffers, and common.do_inference are available.
    
    num_runs = args.num_runs if hasattr(args, 'num_runs') else 10

    # ------------------- PyTorch Inference (CPU) -------------------
    x = torch.randn(args.batch_size, 3, 256, 256)
    # Warm-up run
    _ = torch_model(x)
    torch_times = []
    for _ in range(num_runs):
        start = time.time()
        torch_out_cls, torch_out_seg = torch_model(x)
        torch_times.append(time.time() - start)
    avg_torch_time = sum(torch_times) / num_runs

    # Detailed CPU profiling using PyTorch's autograd profiler:
    with torch.autograd.profiler.profile() as prof:
        _ = torch_model(x)
    print("PyTorch CPU Profiling:")
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    # ------------------- ONNX Runtime Inference (CPU) -------------------
    so = onnxruntime.SessionOptions()
    so.enable_profiling = True  # Enable profiling in ONNX Runtime.
    ort_session = onnxruntime.InferenceSession(args.onnx_model,
                                               sess_options=so,
                                               providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    _ = ort_session.run(None, ort_inputs)  # warm-up run
    ort_times = []
    for _ in range(num_runs):
        start = time.time()
        _ = ort_session.run(None, ort_inputs)
        ort_times.append(time.time() - start)
    avg_ort_time = sum(ort_times) / num_runs
    profile_file = ort_session.end_profiling()
    print(f"ONNX Runtime CPU profiling data saved to: {profile_file}")

    # ------------------- TensorRT Inference (with CPU input) -------------------
    # Note: Although TensorRT runs on the GPU, this test uses CPU inputs.
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(args.trt_model, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        
        try:
            class TRTProfiler(trt.IProfiler):
                def __init__(self):
                    super(TRTProfiler, self).__init__()
                    self.layer_times = {}

                def report_layer_time(self, layer_name, ms):
                    if layer_name in self.layer_times:
                        self.layer_times[layer_name].append(ms)
                    else:
                        self.layer_times[layer_name] = [ms]
            trt_profiler = TRTProfiler()
            context.profiler = trt_profiler
        except Exception as e:
            print("Could not attach custom TRT profiler:", e)
        
        context.set_input_shape("input", (args.batch_size, 3, 256, 256))
        inputs, outputs, bindings, stream = common.allocate_buffers(engine, context=context)
        inputs[0].host = x.numpy()
        
        _ = common.do_inference(context, engine=engine, bindings=bindings,
                                inputs=inputs, outputs=outputs, stream=stream)  # warm-up
        trt_times = []
        for _ in range(num_runs):
            inputs[0].host = x.numpy()
            start = time.time()
            _ = common.do_inference(context, engine=engine, bindings=bindings,
                                    inputs=inputs, outputs=outputs, stream=stream)
            trt_times.append(time.time() - start)
        avg_trt_time = sum(trt_times) / num_runs
        
        # Print TensorRT profiling info if available.
        try:
            if trt_profiler.layer_times:
                print("TensorRT CPU Profiling (per-layer):")
                for layer, times in trt_profiler.layer_times.items():
                    avg_layer_time = sum(times) / len(times)
                    print(f"  {layer}: {avg_layer_time:.3f} ms")
        except Exception as e:
            print("Error retrieving TRT profiling info:", e)
    
    print(f'PyTorch CPU average inference time over {num_runs} runs: {avg_torch_time:.6f} s')
    print(f'ONNX Runtime CPU average inference time over {num_runs} runs: {avg_ort_time:.6f} s')
    print(f'TensorRT (CPU input) average inference time over {num_runs} runs: {avg_trt_time:.6f} s')

    return avg_torch_time, avg_ort_time, avg_trt_time

def inference_speed_test_gpu(torch_model, args):

    # Assume `to_numpy` converts a torch.Tensor to a NumPy array.
    # Also assume that common.allocate_buffers and common.do_inference are defined.
    
    num_runs = args.num_runs if hasattr(args, 'num_runs') else 10

    # ------------------- PyTorch Inference (GPU) -------------------
    x = torch.randn(args.batch_size, 3, 256, 256, device='cuda')
    torch_model = torch_model.cuda()
    
    # Warm-up run
    _ = torch_model(x)
    torch.cuda.synchronize()
    torch_times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        torch_out_cls, torch_out_seg = torch_model(x)
        torch.cuda.synchronize()
        torch_times.append(time.time() - start)
    avg_torch_time = sum(torch_times) / num_runs

    # Detailed profiling using PyTorch's autograd profiler:
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        _ = torch_model(x)
    print("PyTorch GPU Profiling:")
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    
    # ------------------- ONNX Runtime Inference (GPU) -------------------
    so = onnxruntime.SessionOptions()
    so.enable_profiling = True  # Enable detailed profiling in ONNX Runtime
    ort_session = onnxruntime.InferenceSession(args.onnx_model,
                                               sess_options=so,
                                               providers=['CUDAExecutionProvider'])
    # ONNX Runtime expects CPU NumPy arrays as input even when using GPU.
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    _ = ort_session.run(None, ort_inputs)  # warm-up run
    ort_times = []
    for _ in range(num_runs):
        start = time.time()
        _ = ort_session.run(None, ort_inputs)
        ort_times.append(time.time() - start)
    avg_ort_time = sum(ort_times) / num_runs
    profile_file = ort_session.end_profiling()  # Ends profiling and returns the profile file path.
    print(f"ONNX Runtime GPU profiling data saved to: {profile_file}")

    # ------------------- TensorRT Inference (GPU) -------------------
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(args.trt_model, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        # Optionally attach a custom profiler to TensorRT.
        try:
            class TRTProfiler(trt.IProfiler):
                def __init__(self):
                    super(TRTProfiler, self).__init__()
                    self.layer_times = {}

                def report_layer_time(self, layer_name, ms):
                    if layer_name in self.layer_times:
                        self.layer_times[layer_name].append(ms)
                    else:
                        self.layer_times[layer_name] = [ms]

            trt_profiler = TRTProfiler()
            context.profiler = trt_profiler
        except Exception as e:
            print("Could not attach custom TRT profiler:", e)
        
        # Set the static input shape.
        context.set_input_shape("input", (args.batch_size, 3, 256, 256))
        inputs, outputs, bindings, stream = common.allocate_buffers(engine, context=context)
        # TensorRT expects the input as a CPU NumPy array (pinned memory will be used internally)
        inputs[0].host = x.cpu().numpy()
        
        _ = common.do_inference(context, engine=engine, bindings=bindings,
                                inputs=inputs, outputs=outputs, stream=stream)  # warm-up run
        trt_times = []
        for _ in range(num_runs):
            inputs[0].host = x.cpu().numpy()
            start = time.time()
            _ = common.do_inference(context, engine=engine, bindings=bindings,
                                    inputs=inputs, outputs=outputs, stream=stream)
            trt_times.append(time.time() - start)
        avg_trt_time = sum(trt_times) / num_runs

        # Print out TensorRT per-layer profiling info if available.
        try:
            if trt_profiler.layer_times:
                print("TensorRT GPU Profiling (per-layer):")
                for layer, times in trt_profiler.layer_times.items():
                    avg_layer_time = sum(times) / len(times)
                    print(f"  {layer}: {avg_layer_time:.3f} ms")
        except Exception as e:
            print("Error retrieving TRT profiling info:", e)
    
    print(f'PyTorch GPU average inference time over {num_runs} runs: {avg_torch_time:.6f} s')
    print(f'ONNX Runtime GPU average inference time over {num_runs} runs: {avg_ort_time:.6f} s')
    print(f'TensorRT GPU average inference time over {num_runs} runs: {avg_trt_time:.6f} s')

    return avg_torch_time, avg_ort_time, avg_trt_time



    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pytorch_model', type=str, default='model.pth', help='PyTorch model file')
    parser.add_argument('--onnx_model', type=str, default='model.onnx', help='ONNX model file')
    parser.add_argument('--trt_model', type=str, default='model.trt', help='TensorRT model file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument("--backbone", type=str, default="MetaFormer-CAS18-FPN")
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--speed_test_only", type=bool, default=False)
    parser.add_argument("--num_runs", type=int, default=20)
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"])

    args = parser.parse_args()

    args.weights = 'None'

    # Load PyTorch model
    torch_model = MetaFormer.MetaFormerFPN(args)

    # remove model.backbone. from all keys in state_dict
    state_dict = torch.load(args.pytorch_model,  weights_only=True)['state_dict']
    new_state_dict = {}
    for key in state_dict:
        new_state_dict[key.replace('model.backbone.', '')] = state_dict[key]

    # remove cls_criterion.pos_weight from state_dict
    new_state_dict.pop('cls_criterion.pos_weight', None)

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
  
    if not args.speed_test_only:
        onnx_model = export_to_onnx(torch_model, args)
        export_to_trt(args)
        print('Done converting models!')

    # Inference speed test
    # RUN inference speed test with the 3 approaches

    # 1. PyTorch
    # 2. ONNX Runtime
    # 3. TensorRT

    # CPU
    torch_time, ort_time, trt_time = inference_speed_test_cpu(torch_model, args)
    # GPU
    torch_time, ort_time, trt_time = inference_speed_test_gpu(torch_model, args)





   


