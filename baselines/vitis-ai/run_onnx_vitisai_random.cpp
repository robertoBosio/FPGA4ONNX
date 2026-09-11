#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/vitisai/vitisai_provider_factory.h>

static void check_status(OrtStatus* status) {
  if (status != nullptr) {
    const char* msg = Ort::GetApi().GetErrorMessage(status);
    std::cerr << msg << std::endl;
    Ort::GetApi().ReleaseStatus(status);
    std::exit(1);
  }
}

static int64_t product(const std::vector<int64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

int main(int argc, char** argv) {
  if (argc < 2 || argc > 6) {
    std::cerr << "usage: " << argv[0]
              << " <model.onnx> [preprocessed_input.bin] [output.bin] [iters] [threads]" << std::endl;
    return 2;
  }

  const std::string model = argv[1];
  const bool have_input_arg = (argc >= 3);
  const std::string input_arg = have_input_arg ? argv[2] : "";
  const bool use_random_input = !have_input_arg || input_arg == "-" || input_arg == "random";
  const bool have_input_file = have_input_arg && !use_random_input;
  const std::string output_file = (argc >= 4) ? argv[3] : "";
  const int iters = (argc >= 5) ? std::atoi(argv[4]) : 50;
  const int threads = (argc >= 6) ? std::atoi(argv[5]) : 1;
  const int warmup = 5;
  if (iters <= 0 || threads <= 0) {
    std::cerr << "iters and threads must be positive" << std::endl;
    return 2;
  }
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo_random_vitisai");
  Ort::SessionOptions options;
  check_status(OrtSessionOptionsAppendExecutionProvider_VITISAI(options, ""));

  std::cout << "creating session: " << model << std::endl;
  Ort::Session session(env, model.c_str(), options);
  Ort::AllocatorWithDefaultOptions allocator;

  auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
  const char* input_name = input_name_alloc.get();
  auto input_info = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
  std::vector<int64_t> input_shape;
  try {
    input_shape = input_info.GetShape();
    for (auto& dim : input_shape) if (dim <= 0) dim = 1;
  } catch (...) {
    std::cerr << "WARN: input GetShape failed, fallback [1,3,640,640]" << std::endl;
    input_shape = {1, 3, 640, 640};
  }

  const auto input_elements = product(input_shape);
  std::vector<float> input(input_elements);
  if (have_input_file) {
    std::ifstream f(input_arg, std::ios::binary);
    if (!f) { std::cerr << "cannot open input file " << input_arg << std::endl; return 3; }
    f.read(reinterpret_cast<char*>(input.data()),
           static_cast<std::streamsize>(input.size() * sizeof(float)));
    if (!f && !f.eof()) { std::cerr << "input file read error" << std::endl; return 3; }
    std::cout << "loaded preprocessed input from " << input_arg << std::endl;
  } else {
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::generate(input.begin(), input.end(), [&] { return dist(gen); });
    std::cout << "generated deterministic random input with seed 0" << std::endl;
  }

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input.data(), input.size(), input_shape.data(), input_shape.size());

  std::vector<Ort::AllocatedStringPtr> output_name_allocs;
  std::vector<const char*> output_names;
  const auto output_count = session.GetOutputCount();
  output_name_allocs.reserve(output_count);
  output_names.reserve(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    output_name_allocs.emplace_back(session.GetOutputNameAllocated(i, allocator));
    output_names.push_back(output_name_allocs.back().get());
  }

  std::cout << "input: " << input_name;
  for (auto dim : input_shape) std::cout << " " << dim;
  std::cout << std::endl;

  std::cout << "input: " << input_name;
  for (auto dim : input_shape) std::cout << " " << dim;
  std::cout << std::endl;

  using clock = std::chrono::high_resolution_clock;

  auto run_once = [&]() {
    return session.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1,
                       output_names.data(), output_names.size());
  };

  for (int w = 0; w < warmup; ++w) {
    auto o = run_once();
    (void)o;
  }

  if (threads == 1) {
    std::cout << "running " << iters << " iterations (warmup " << warmup << ")" << std::endl;
  } else {
    std::cout << "running " << threads << " threads x " << iters
              << " iterations (warmup " << warmup << ")" << std::endl;
  }

  std::vector<std::vector<double>> thread_times(threads);
  for (auto& v : thread_times) v.reserve(iters);

  auto t_start = clock::now();
  std::vector<std::thread> workers;
  workers.reserve(threads);
  for (int tid = 0; tid < threads; ++tid) {
    workers.emplace_back([&, tid]() {
      for (int k = 0; k < iters; ++k) {
        auto t0 = clock::now();
        auto outputs = run_once();
        auto t1 = clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        thread_times[tid].push_back(ms);
        if (tid == 0 && k == 0 && !output_file.empty() && output_file != "/dev/null" && outputs.size() > 0) {
          auto info = outputs[0].GetTensorTypeAndShapeInfo();
          std::vector<int64_t> shape;
          try { shape = info.GetShape(); } catch (...) { shape = {1, 84, 8400}; }
          auto* data = outputs[0].GetTensorMutableData<float>();
          const auto count = product(shape);
          std::ofstream f(output_file, std::ios::binary);
          f.write(reinterpret_cast<const char*>(data),
                  static_cast<std::streamsize>(count * sizeof(float)));
          std::cout << "wrote output " << count << " floats to " << output_file << std::endl;
        }
      }
    });
  }
  for (auto& worker : workers) worker.join();
  auto t_end = clock::now();

  std::vector<double> times;
  times.reserve(static_cast<size_t>(iters) * threads);
  for (const auto& v : thread_times) times.insert(times.end(), v.begin(), v.end());

  double sum = std::accumulate(times.begin(), times.end(), 0.0);
  double avg = sum / times.size();
  double mn = *std::min_element(times.begin(), times.end());
  double mx = *std::max_element(times.begin(), times.end());
  std::sort(times.begin(), times.end());
  double med = times[times.size() / 2];
  double p90 = times[static_cast<size_t>(times.size() * 0.9)];
  std::cout << "LATENCY ms: avg " << avg << " median " << med
            << " p90 " << p90 << " min " << mn << " max " << mx << std::endl;
  const double wall_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  const int total = iters * threads;
  std::cout << "WALL ms: " << wall_ms << " total_inferences " << total << std::endl;
  std::cout << "THROUGHPUT inf/s: " << (1000.0 * total / wall_ms) << std::endl;

  return 0;
}
