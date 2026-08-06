// TensorRT 10.x FireNet runtime + ONNX builder.
// Compile only when TensorRT is found (LIBEVENTGATE_HAS_TRT).

#include "libeventgate/firenet_trt.hpp"
#include "libeventgate/vram.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace eventgate {
namespace {

class TrtLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TRT] " << msg << "\n";
    }
  }
};

TrtLogger g_logger;

size_t dtype_size(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF:  return 2;
    case nvinfer1::DataType::kINT8:  return 1;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kBOOL:  return 1;
    default: return 4;
  }
}

size_t volume(const nvinfer1::Dims& d) {
  size_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) {
    // TRT 10 Dims::d is int64_t — keep both args same type for std::max
    const int64_t di = std::max<int64_t>(d.d[i], 1);
    v *= static_cast<size_t>(di);
  }
  return v;
}

// Sort key: trailing digits in h_in_0 / h_out_12, else name.
int state_order_key(const std::string& name) {
  static const std::regex re(R"((\d+)\s*$)");
  std::smatch m;
  if (std::regex_search(name, m, re)) return std::stoi(m[1].str());
  return 0;
}

bool name_has(const std::string& n, const char* needle) {
  return n.find(needle) != std::string::npos;
}

bool is_state_name(const std::string& ns) {
  return name_has(ns, "h_in") || name_has(ns, "h_out") || name_has(ns, "state");
}

bool is_voxel_name(const std::string& ns) {
  return name_has(ns, "voxel") || ns == "input" || ns == "events" || ns == "event_tensor";
}

bool is_frame_name(const std::string& ns) {
  return name_has(ns, "frame") || ns == "output" || ns == "image" || name_has(ns, "img");
}

// Device: cast __half → float on GPU for D2H-free half frame paths when needed.
// Simple kernel-free approach: keep pinned half buffer + host convert after stream sync.
// Prefer engine FP32 output when possible; still handle half.

} // namespace

struct FireNetTrt::Impl {
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;

  struct Buf {
    std::string name;
    bool is_input = false;
    bool is_state = false;
    bool skip_own_alloc = false; // external voxel pointer each frame
    void* device = nullptr;
    size_t bytes = 0;
    size_t elems = 0;
    nvinfer1::DataType dtype{};
    nvinfer1::Dims dims{};
  };
  std::vector<Buf> bindings;
  int voxel_index = -1;
  int frame_index = -1;
  std::vector<int> h_in_indices;  // sorted by order key
  std::vector<int> h_out_indices;
  std::vector<void*> state_hold; // device copies of last committed h_out

  // Pinned half staging for async D2H when frame tensor is FP16
  __half*  h_frame_f16 = nullptr;
  size_t   h_frame_elems = 0;
  bool warned_no_state = false;
};

FireNetTrt::FireNetTrt(const std::string& engine_path) : impl_(std::make_unique<Impl>()) {
  std::ifstream f(engine_path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open engine: " + engine_path);
  f.seekg(0, std::ios::end);
  const size_t sz = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  std::vector<char> blob(sz);
  f.read(blob.data(), static_cast<std::streamsize>(sz));

  impl_->runtime.reset(nvinfer1::createInferRuntime(g_logger));
  impl_->engine.reset(impl_->runtime->deserializeCudaEngine(blob.data(), blob.size()));
  if (!impl_->engine) throw std::runtime_error("deserializeCudaEngine failed");
  impl_->context.reset(impl_->engine->createExecutionContext());
  if (!impl_->context) throw std::runtime_error("createExecutionContext failed");

  const int n = impl_->engine->getNbIOTensors();
  for (int i = 0; i < n; ++i) {
    const char* name = impl_->engine->getIOTensorName(i);
    Impl::Buf b;
    b.name = name;
    b.is_input = impl_->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
    b.dtype = impl_->engine->getTensorDataType(name);
    b.dims = impl_->engine->getTensorShape(name);
    // Resolve -1 (dynamic) dims to 1 for byte sizing fallback — user builds fixed graph.
    for (int d = 0; d < b.dims.nbDims; ++d) {
      if (b.dims.d[d] < 0) b.dims.d[d] = 1;
    }
    b.elems = volume(b.dims);
    b.bytes = b.elems * dtype_size(b.dtype);
    b.is_state = is_state_name(b.name);

    // Voxel: do not own a binding buffer — pipeline passes device volume each call.
    // Keep a small fallback alloc only if caller passes null (debug).
    if (is_voxel_name(b.name) && b.is_input) {
      b.skip_own_alloc = true;
      if (b.bytes > 0) {
        cudaError_t e = cudaMalloc(&b.device, b.bytes);
        if (e != cudaSuccess) throw std::runtime_error("cudaMalloc fallback voxel " + b.name);
        device_bytes_ += b.bytes;
        cudaMemset(b.device, 0, b.bytes);
      }
      impl_->voxel_index = static_cast<int>(impl_->bindings.size());
      voxel_fp16_ = (b.dtype == nvinfer1::DataType::kHALF);
      voxel_bytes_ = b.bytes;
    } else if (b.bytes > 0) {
      cudaError_t e = cudaMalloc(&b.device, b.bytes);
      if (e != cudaSuccess) throw std::runtime_error("cudaMalloc binding " + b.name);
      device_bytes_ += b.bytes;
      cudaMemset(b.device, 0, b.bytes);
    }

    if (is_frame_name(b.name) && !b.is_input) {
      impl_->frame_index = static_cast<int>(impl_->bindings.size());
    }
    if (b.is_state && b.is_input)
      impl_->h_in_indices.push_back(static_cast<int>(impl_->bindings.size()));
    if (b.is_state && !b.is_input)
      impl_->h_out_indices.push_back(static_cast<int>(impl_->bindings.size()));

    impl_->bindings.push_back(b);
  }

  // Fallback name matching
  if (impl_->voxel_index < 0) {
    for (int i = 0; i < static_cast<int>(impl_->bindings.size()); ++i) {
      if (impl_->bindings[static_cast<size_t>(i)].is_input &&
          !impl_->bindings[static_cast<size_t>(i)].is_state) {
        impl_->voxel_index = i;
        auto& b = impl_->bindings[static_cast<size_t>(i)];
        voxel_fp16_ = (b.dtype == nvinfer1::DataType::kHALF);
        voxel_bytes_ = b.bytes;
        break;
      }
    }
  }
  if (impl_->frame_index < 0) {
    for (int i = 0; i < static_cast<int>(impl_->bindings.size()); ++i) {
      if (!impl_->bindings[static_cast<size_t>(i)].is_input &&
          !impl_->bindings[static_cast<size_t>(i)].is_state) {
        impl_->frame_index = i;
        break;
      }
    }
  }

  // Stable pairing: sort by numeric suffix so h_in_0 ↔ h_out_0
  auto by_key = [&](int a, int b) {
    return state_order_key(impl_->bindings[static_cast<size_t>(a)].name) <
           state_order_key(impl_->bindings[static_cast<size_t>(b)].name);
  };
  std::sort(impl_->h_in_indices.begin(), impl_->h_in_indices.end(), by_key);
  std::sort(impl_->h_out_indices.begin(), impl_->h_out_indices.end(), by_key);

  has_state_ = !impl_->h_in_indices.empty() && !impl_->h_out_indices.empty() &&
               impl_->h_in_indices.size() == impl_->h_out_indices.size();
  n_state_pairs_ = has_state_ ? static_cast<int>(impl_->h_in_indices.size()) : 0;

  if (has_state_) {
    for (int idx : impl_->h_out_indices) {
      void* p = nullptr;
      const size_t bytes = impl_->bindings[static_cast<size_t>(idx)].bytes;
      cudaMalloc(&p, bytes);
      device_bytes_ += bytes;
      cudaMemcpy(p, impl_->bindings[static_cast<size_t>(idx)].device, bytes,
                 cudaMemcpyDeviceToDevice);
      impl_->state_hold.push_back(p);
    }
  }

  // Pinned host half buffer when frame binding is FP16
  if (impl_->frame_index >= 0) {
    const auto& fr = impl_->bindings[static_cast<size_t>(impl_->frame_index)];
    impl_->h_frame_elems = fr.elems;
    if (fr.dtype == nvinfer1::DataType::kHALF) {
      cudaHostAlloc(&impl_->h_frame_f16, fr.elems * sizeof(__half), cudaHostAllocDefault);
    }
  }

  // Soft VRAM budget estimate for bindings alone (engine runtime adds workspace at build)
  const float bind_mib = static_cast<float>(device_bytes_) / (1024.f * 1024.f);
  std::cout << "FireNetTrt: tensors=" << n
            << " voxel_idx=" << impl_->voxel_index
            << " frame_idx=" << impl_->frame_index
            << " state_pairs=" << n_state_pairs_
            << " voxel_fp16=" << voxel_fp16_
            << " bindings_mib=" << bind_mib << "\n";
  if (has_state_) {
    for (size_t k = 0; k < impl_->h_in_indices.size(); ++k) {
      const auto& in = impl_->bindings[static_cast<size_t>(impl_->h_in_indices[k])];
      const auto& out = impl_->bindings[static_cast<size_t>(impl_->h_out_indices[k])];
      std::cout << "  state[" << k << "] " << in.name << " ↔ " << out.name
                << "  bytes=" << in.bytes << "\n";
    }
  }

  // FireNet at 640x512 with two ConvGRU states is ~tens of MiB — flag anything runaway.
  if (bind_mib > 1500.f) {
    std::cerr << "[vram] WARN: FireNet bindings alone " << bind_mib
              << " MiB — unexpected for plain FireNet; check ONNX shapes\n";
  }
}

FireNetTrt::~FireNetTrt() {
  if (!impl_) return;
  for (auto& b : impl_->bindings) {
    if (b.device) cudaFree(b.device);
  }
  for (void* p : impl_->state_hold) {
    if (p) cudaFree(p);
  }
  if (impl_->h_frame_f16) cudaFreeHost(impl_->h_frame_f16);
}

void FireNetTrt::run_frame(const void* voxel_device,
                           float* out_frame_host,
                           int height,
                           int width,
                           bool freeze_state,
                           cudaStream_t stream) {
  if (impl_->voxel_index < 0 || impl_->frame_index < 0) {
    throw std::runtime_error("FireNetTrt: missing voxel/frame bindings");
  }

  auto& vox = impl_->bindings[static_cast<size_t>(impl_->voxel_index)];
  void* voxel_ptr = const_cast<void*>(voxel_device);
  if (!voxel_ptr) voxel_ptr = vox.device;
  if (!voxel_ptr) throw std::runtime_error("FireNetTrt: null voxel pointer");

  impl_->context->setTensorAddress(vox.name.c_str(), voxel_ptr);

  for (size_t i = 0; i < impl_->bindings.size(); ++i) {
    if (static_cast<int>(i) == impl_->voxel_index) continue;
    auto& b = impl_->bindings[i];
    if (!b.device) throw std::runtime_error("null binding device: " + b.name);
    impl_->context->setTensorAddress(b.name.c_str(), b.device);
  }

  if (!impl_->context->enqueueV3(stream)) {
    throw std::runtime_error("enqueueV3 failed");
  }

  // State write-back control for IMU gate
  if (has_state_) {
    if (freeze_state) {
      // Ignore network h_out; re-seed h_in from held commit.
      for (size_t k = 0; k < impl_->h_in_indices.size() && k < impl_->state_hold.size(); ++k) {
        const int in_i = impl_->h_in_indices[k];
        const size_t bytes = impl_->bindings[static_cast<size_t>(in_i)].bytes;
        cudaMemcpyAsync(impl_->bindings[static_cast<size_t>(in_i)].device, impl_->state_hold[k],
                        bytes, cudaMemcpyDeviceToDevice, stream);
      }
    } else {
      // Commit: h_out → h_in and refresh hold
      const size_t n_pair = std::min(impl_->h_in_indices.size(), impl_->h_out_indices.size());
      for (size_t k = 0; k < n_pair; ++k) {
        const int in_i = impl_->h_in_indices[k];
        const int out_i = impl_->h_out_indices[k];
        const size_t bytes = impl_->bindings[static_cast<size_t>(in_i)].bytes;
        cudaMemcpyAsync(impl_->bindings[static_cast<size_t>(in_i)].device,
                        impl_->bindings[static_cast<size_t>(out_i)].device, bytes,
                        cudaMemcpyDeviceToDevice, stream);
        if (k < impl_->state_hold.size()) {
          cudaMemcpyAsync(impl_->state_hold[k], impl_->bindings[static_cast<size_t>(out_i)].device,
                          bytes, cudaMemcpyDeviceToDevice, stream);
        }
      }
    }
  } else if (freeze_state && !impl_->warned_no_state) {
    std::cerr << "[gate] WARN: engine has no state I/O — freeze is a no-op until ONNX export "
                 "externalizes ConvGRU h_t (re-run export without --no-externalize-state)\n";
    impl_->warned_no_state = true;
  }

  auto& fr = impl_->bindings[static_cast<size_t>(impl_->frame_index)];
  const size_t hw = static_cast<size_t>(height) * static_cast<size_t>(width);
  // Frame may be [1,1,H,W] or [1,H,W] or [H,W] — take last height*width elements if larger
  if (fr.elems < hw) {
    throw std::runtime_error("frame tensor elems (" + std::to_string(fr.elems) +
                             ") < height*width (" + std::to_string(hw) + ")");
  }
  const size_t offset_elems = fr.elems - hw; // if NCHW with leading ones, last plane is image

  if (fr.dtype == nvinfer1::DataType::kFLOAT) {
    // Prefer caller's host buffer if large enough; otherwise pin-staging then memcpy.
    float* dst = out_frame_host;
    void* d_src = static_cast<char*>(fr.device) + offset_elems * sizeof(float);
    cudaMemcpyAsync(dst, d_src, hw * sizeof(float), cudaMemcpyDeviceToHost, stream);
  } else if (fr.dtype == nvinfer1::DataType::kHALF) {
    if (!impl_->h_frame_f16) {
      throw std::runtime_error("missing pinned half frame buffer");
    }
    void* d_src = static_cast<char*>(fr.device) + offset_elems * sizeof(__half);
    cudaMemcpyAsync(impl_->h_frame_f16, d_src, hw * sizeof(__half), cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);
    for (size_t i = 0; i < hw; ++i) {
      out_frame_host[i] = __half2float(impl_->h_frame_f16[i]);
    }
  } else {
    throw std::runtime_error("unsupported frame dtype");
  }
}

bool build_engine_from_onnx(const std::string& onnx_path,
                            const std::string& engine_out,
                            const PipelineConfig& cfg,
                            std::string* err) {
  try {
    log_vram("build_engine_start", cfg.vram_warn_mib);

    // Workspace cap is the main OOM lever on 6 GB during build (builder peaks higher than runtime).
    if (cfg.trt_workspace_bytes > 512ull * 1024ull * 1024ull) {
      std::cerr << "[build_engine] WARN: workspace > 512 MiB requested ("
                << (cfg.trt_workspace_bytes / (1024 * 1024))
                << " MiB). Soft budget soft-flag — may OOM on 6 GB during build.\n";
    }

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    // TRT 10+: createNetworkV2(0) is the supported path (explicit batch flag is deprecated).
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto parser =
        std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, g_logger));
    if (!parser) {
      if (err)
        *err =
            "nvonnxparser::createParser failed — missing libnvonnxparser.so (incomplete TRT install)";
      return false;
    }
    if (!parser->parseFromFile(onnx_path.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
      if (err) *err = "ONNX parse failed — check FireNet ConvGRU ops / opset";
      return false;
    }

    std::cout << "network: " << network->getNbInputs() << " inputs, " << network->getNbOutputs()
              << " outputs\n";
    for (int i = 0; i < network->getNbInputs(); ++i) {
      auto* t = network->getInput(i);
      std::cout << "  in  " << t->getName() << "\n";
    }
    for (int i = 0; i < network->getNbOutputs(); ++i) {
      auto* t = network->getOutput(i);
      std::cout << "  out " << t->getName() << "\n";
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, cfg.trt_workspace_bytes);

    // Prefer strong typing API over deprecated BuilderFlag::kFP16 when available.
    bool use_fp16 = cfg.prefer_fp16 || cfg.prefer_int8;
    if (cfg.prefer_int8) {
      std::cerr << "[build_engine] INT8 preferred but no calibrator yet — using FP16. "
                   "Add calibrator after first real session for INT8.\n";
    }
    if (use_fp16) {
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
      std::cout << "builder: FP16 enabled, workspace_mb="
                << (cfg.trt_workspace_bytes / (1024 * 1024)) << "\n";
    } else {
      std::cout << "builder: FP32, workspace_mb=" << (cfg.trt_workspace_bytes / (1024 * 1024))
                << "\n";
    }

    log_vram("build_engine_before_build", cfg.vram_warn_mib);
    auto serialized =
        std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
      if (err)
        *err =
            "buildSerializedNetwork failed (check VRAM; keep --workspace-mb 512; close other GPU apps)";
      return false;
    }

    std::ofstream out(engine_out, std::ios::binary);
    if (!out) {
      if (err) *err = "cannot write " + engine_out;
      return false;
    }
    out.write(static_cast<const char*>(serialized->data()),
              static_cast<std::streamsize>(serialized->size()));

    log_vram("build_engine_end", cfg.vram_warn_mib);
    std::cout << "wrote engine " << engine_out << " (" << serialized->size() << " bytes)\n"
              << "NOTE: builder peak VRAM is often higher than runtime; runtime FireNet on "
                 "640x512 is small (<<1 GiB bindings + activations).\n";
    return true;
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return false;
  }
}

} // namespace eventgate
