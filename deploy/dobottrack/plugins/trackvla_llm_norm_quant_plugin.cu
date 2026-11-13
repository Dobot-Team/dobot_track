// Custom TensorRT-LLM RMSNorm + dynamic INT8 quantization kernels.
// Interface-compatible with the TRT-LLM 0.12 SmoothQuant plugins.
// Build on the target Orin: deploy/dobottrack/build_plugin.sh.
// This plugin is for the LLM; Vision uses TensorRT-native implementations.

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace trackvla_llm
{

constexpr char const* kNamespace{"TrackVLALLM"};
constexpr char const* kRmsnormQuantName{"TrackVLARmsnormQuant"};
constexpr char const* kQuantizePerTokenName{"TrackVLAQuantizePerToken"};
constexpr char const* kPluginVersion{"1"};
constexpr int32_t kMaxWarpVpl{16};
constexpr int32_t kRmsnormWarpsPerBlock{4};
constexpr int32_t kQuantWarpsPerBlock{4};
constexpr size_t kMaxSmemBytes{48u * 1024u};

// -------------------------------------------------------------------------------------------
// Device helpers.  cvt_rni_sat_s8 mirrors tensorrt_llm::common::cuda_cast<int8_t, float>,
// which lowers to cvt.rni.sat.s8.f32 (round to nearest even, saturating).
// -------------------------------------------------------------------------------------------

__device__ __forceinline__ signed char cvt_rni_sat_s8(float value)
{
    union
    {
        signed char i8[2];
        short i16;
    } out;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=h"(out.i16) : "f"(value));
    return out.i8[0];
}

__device__ __forceinline__ float warp_reduce_sum(float value)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float warp_reduce_max(float value)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float block_reduce_max(float value, float* scratch, float* broadcast)
{
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    int const warps = ((int) blockDim.x + 31) >> 5;
    value = warp_reduce_max(value);
    if (lane == 0)
    {
        scratch[warp] = value;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float total = -INFINITY;
        for (int index = 0; index < warps; ++index)
        {
            total = fmaxf(total, scratch[index]);
        }
        *broadcast = total;
    }
    __syncthreads();
    return *broadcast;
}

// -------------------------------------------------------------------------------------------
// Fast path: fused RMSNorm + per-token INT8 quant, one warp per row, hidden == 64 * VPL.
//
// Reproduces tensorrt_llm::kernels::generalRmsNorm<half2, int8, true>:
//   rstd  = rsqrt(mean(x^2) + eps)
//   y     = half(x * rstd * gamma + beta)
//   scale = 127 / max|y|;  q = cvt.rni.sat.s8(y * scale);  scale_row = max|y| / 127
// The max is taken over the half-rounded y, and the floor on |y| is 1e-6, as in the
// reference. The sum-of-squares association differs from the CTA-wide tree, so rstd can
// differ in the last ulp and affect INT8 rounding near quantization boundaries.
// -------------------------------------------------------------------------------------------

template <int VPL, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(32 * WARPS_PER_BLOCK) void warp_rmsnorm_quant_kernel(
    __half const* __restrict__ input, __half const* __restrict__ gamma,
    __half const* __restrict__ beta, float eps, signed char* __restrict__ quantized,
    float* __restrict__ scale_out, int rows, int hidden)
{
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    int const row = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= rows)
    {
        return;
    }

    __half2 const* row_in = reinterpret_cast<__half2 const*>(input + (size_t) row * (size_t) hidden);
    __half2 const* gamma2 = reinterpret_cast<__half2 const*>(gamma);
    __half2 const* beta2 = reinterpret_cast<__half2 const*>(beta);

    __half2 values[VPL];
    float sum_squares = 0.0f;
#pragma unroll
    for (int k = 0; k < VPL; ++k)
    {
        values[k] = row_in[lane + 32 * k];
        float2 const as_float = __half22float2(values[k]);
        sum_squares = fmaf(as_float.x, as_float.x, sum_squares);
        sum_squares = fmaf(as_float.y, as_float.y, sum_squares);
    }
    float const rstd = rsqrtf(warp_reduce_sum(sum_squares) / (float) hidden + eps);

    __half2 normalized[VPL];
    float amax = 1e-6f;
#pragma unroll
    for (int k = 0; k < VPL; ++k)
    {
        float2 const val = __half22float2(values[k]);
        float2 const gain = __half22float2(gamma2[lane + 32 * k]);
        float2 y;
        y.x = val.x * rstd * gain.x;
        y.y = val.y * rstd * gain.y;
        if (beta2 != nullptr)
        {
            float2 const bias = __half22float2(beta2[lane + 32 * k]);
            y.x += bias.x;
            y.y += bias.y;
        }
        normalized[k] = __floats2half2_rn(y.x, y.y);
        float2 const rounded = __half22float2(normalized[k]);
        amax = fmaxf(amax, fmaxf(fabsf(rounded.x), fabsf(rounded.y)));
    }
    float const abs_max = warp_reduce_max(amax);
    float const dynamic_scale = 127.0f / abs_max;

    char2* row_out = reinterpret_cast<char2*>(quantized + (size_t) row * (size_t) hidden);
#pragma unroll
    for (int k = 0; k < VPL; ++k)
    {
        float2 const rounded = __half22float2(normalized[k]);
        char2 out;
        out.x = cvt_rni_sat_s8(rounded.x * dynamic_scale);
        out.y = cvt_rni_sat_s8(rounded.y * dynamic_scale);
        row_out[lane + 32 * k] = out;
    }
    if (lane == 0)
    {
        scale_out[row] = abs_max / 127.0f;
    }
}

// -------------------------------------------------------------------------------------------
// Generic fused path: same maths, one warp per row, three streaming passes over the row and
// no shared memory. Valid for any hidden size; the row stays resident in L2.
// -------------------------------------------------------------------------------------------

template <int WARPS_PER_BLOCK>
__global__ __launch_bounds__(32 * WARPS_PER_BLOCK) void warp_rmsnorm_quant_stream_kernel(
    __half const* __restrict__ input, __half const* __restrict__ gamma,
    __half const* __restrict__ beta, float eps, signed char* __restrict__ quantized,
    float* __restrict__ scale_out, int rows, int hidden)
{
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    int const row = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= rows)
    {
        return;
    }

    __half const* row_in = input + (size_t) row * (size_t) hidden;

    float sum_squares = 0.0f;
    for (int i = lane; i < hidden; i += 32)
    {
        float const value = __half2float(row_in[i]);
        sum_squares = fmaf(value, value, sum_squares);
    }
    float const rstd = rsqrtf(warp_reduce_sum(sum_squares) / (float) hidden + eps);

    float amax = 1e-6f;
    for (int i = lane; i < hidden; i += 32)
    {
        float y = __half2float(row_in[i]) * rstd * __half2float(gamma[i]);
        if (beta != nullptr)
        {
            y += __half2float(beta[i]);
        }
        amax = fmaxf(amax, fabsf(__half2float(__float2half_rn(y))));
    }
    float const abs_max = warp_reduce_max(amax);
    float const dynamic_scale = 127.0f / abs_max;

    signed char* row_out = quantized + (size_t) row * (size_t) hidden;
    for (int i = lane; i < hidden; i += 32)
    {
        float y = __half2float(row_in[i]) * rstd * __half2float(gamma[i]);
        if (beta != nullptr)
        {
            y += __half2float(beta[i]);
        }
        y = __half2float(__float2half_rn(y));
        row_out[i] = cvt_rni_sat_s8(y * dynamic_scale);
    }
    if (lane == 0)
    {
        scale_out[row] = abs_max / 127.0f;
    }
}

// -------------------------------------------------------------------------------------------
// Fast path: per-token INT8 quant, one warp per row, cols == 64 * VPL, registers only.
// Reproduces tensorrt_llm::kernels::perTokenQuantization<half, int8, true>.
// -------------------------------------------------------------------------------------------

template <int VPL, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(32 * WARPS_PER_BLOCK) void warp_pertoken_quant_kernel(
    __half const* __restrict__ input, signed char* __restrict__ quantized,
    float* __restrict__ scale_out, int rows, int cols)
{
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    int const row = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= rows)
    {
        return;
    }

    __half2 const* row_in = reinterpret_cast<__half2 const*>(input + (size_t) row * (size_t) cols);
    __half2 values[VPL];
    float amax = 1e-6f;
#pragma unroll
    for (int k = 0; k < VPL; ++k)
    {
        values[k] = row_in[lane + 32 * k];
        float2 const as_float = __half22float2(values[k]);
        amax = fmaxf(amax, fmaxf(fabsf(as_float.x), fabsf(as_float.y)));
    }
    float const row_max = warp_reduce_max(amax);
    float const dynamic_scale = 127.0f / row_max;

    char2* row_out = reinterpret_cast<char2*>(quantized + (size_t) row * (size_t) cols);
#pragma unroll
    for (int k = 0; k < VPL; ++k)
    {
        float2 const val = __half22float2(values[k]);
        char2 out;
        out.x = cvt_rni_sat_s8(val.x * dynamic_scale);
        out.y = cvt_rni_sat_s8(val.y * dynamic_scale);
        row_out[lane + 32 * k] = out;
    }
    if (lane == 0)
    {
        scale_out[row] = row_max / 127.0f;
    }
}

// -------------------------------------------------------------------------------------------
// Wide-row path: per-token INT8 quant, one CTA per row, half2-vectorised shared-memory row
// cache. Numerically bit identical to the warp path for the same row.
// -------------------------------------------------------------------------------------------

template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void block_pertoken_quant_h2_kernel(
    __half const* __restrict__ input, signed char* __restrict__ quantized,
    float* __restrict__ scale_out, int rows, int cols)
{
    extern __shared__ __align__(16) char smem_raw[];
    __half2* smem = reinterpret_cast<__half2*>(smem_raw);
    __shared__ float scratch[32];
    __shared__ float broadcast;

    int const row = blockIdx.x;
    int const packed = cols / 2;
    __half2 const* row_in = reinterpret_cast<__half2 const*>(input + (size_t) row * (size_t) cols);

    float amax = 1e-6f;
    for (int i = threadIdx.x; i < packed; i += BLOCK)
    {
        __half2 const vec = row_in[i];
        smem[i] = vec;
        float2 const val = __half22float2(vec);
        amax = fmaxf(amax, fmaxf(fabsf(val.x), fabsf(val.y)));
    }
    float const row_max = block_reduce_max(amax, scratch, &broadcast);
    float const dynamic_scale = 127.0f / row_max;

    char2* row_out = reinterpret_cast<char2*>(quantized + (size_t) row * (size_t) cols);
    for (int i = threadIdx.x; i < packed; i += BLOCK)
    {
        float2 const val = __half22float2(smem[i]);
        char2 out;
        out.x = cvt_rni_sat_s8(val.x * dynamic_scale);
        out.y = cvt_rni_sat_s8(val.y * dynamic_scale);
        row_out[i] = out;
    }
    if (threadIdx.x == 0)
    {
        scale_out[row] = row_max / 127.0f;
    }
}

// -------------------------------------------------------------------------------------------
// Generic quant path: one warp per row, two streaming passes, any row width.
// -------------------------------------------------------------------------------------------

template <int WARPS_PER_BLOCK>
__global__ __launch_bounds__(32 * WARPS_PER_BLOCK) void warp_pertoken_quant_stream_kernel(
    __half const* __restrict__ input, signed char* __restrict__ quantized,
    float* __restrict__ scale_out, int rows, int cols)
{
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    int const row = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= rows)
    {
        return;
    }

    __half const* row_in = input + (size_t) row * (size_t) cols;
    float amax = 1e-6f;
    for (int i = lane; i < cols; i += 32)
    {
        amax = fmaxf(amax, fabsf(__half2float(row_in[i])));
    }
    float const row_max = warp_reduce_max(amax);
    float const dynamic_scale = 127.0f / row_max;

    signed char* row_out = quantized + (size_t) row * (size_t) cols;
    for (int i = lane; i < cols; i += 32)
    {
        row_out[i] = cvt_rni_sat_s8(__half2float(row_in[i]) * dynamic_scale);
    }
    if (lane == 0)
    {
        scale_out[row] = row_max / 127.0f;
    }
}

// -------------------------------------------------------------------------------------------
// Host-side launch selection
// -------------------------------------------------------------------------------------------

template <int VPL>
void launch_warp_rmsnorm_quant(__half const* input, __half const* gamma, __half const* beta,
    float eps, signed char* quantized, float* scale_out, int rows, int hidden, cudaStream_t stream)
{
    constexpr int32_t WARPS = kRmsnormWarpsPerBlock;
    int const blocks = (rows + WARPS - 1) / WARPS;
    warp_rmsnorm_quant_kernel<VPL, WARPS>
        <<<blocks, 32 * WARPS, 0, stream>>>(input, gamma, beta, eps, quantized, scale_out, rows, hidden);
}

template <int VPL>
void launch_warp_pertoken_quant(__half const* input, signed char* quantized, float* scale_out,
    int rows, int cols, cudaStream_t stream)
{
    constexpr int32_t WARPS = kQuantWarpsPerBlock;
    int const blocks = (rows + WARPS - 1) / WARPS;
    warp_pertoken_quant_kernel<VPL, WARPS>
        <<<blocks, 32 * WARPS, 0, stream>>>(input, quantized, scale_out, rows, cols);
}

int32_t warp_vpl_for(int hidden)
{
    if (hidden <= 0 || hidden % 64 != 0)
    {
        return 0;
    }
    int32_t const vpl = hidden / 64;
    return (vpl >= 1 && vpl <= kMaxWarpVpl) ? vpl : 0;
}

cudaError_t launch_rmsnorm_quant(__half const* input, __half const* gamma, __half const* beta,
    float eps, signed char* quantized, float* scale_out, int rows, int hidden, cudaStream_t stream)
{
    if (rows <= 0 || hidden <= 0)
    {
        return cudaSuccess;
    }
    switch (warp_vpl_for(hidden))
    {
    case 1: launch_warp_rmsnorm_quant<1>(input, gamma, beta, eps, quantized, scale_out, rows, hidden, stream); break;
    case 2: launch_warp_rmsnorm_quant<2>(input, gamma, beta, eps, quantized, scale_out, rows, hidden, stream); break;
    case 3: launch_warp_rmsnorm_quant<3>(input, gamma, beta, eps, quantized, scale_out, rows, hidden, stream); break;
    case 4: launch_warp_rmsnorm_quant<4>(input, gamma, beta, eps, quantized, scale_out, rows, hidden, stream); break;
    case 5: launch_warp_rmsnorm_quant<5>(input, gamma, beta, eps, quantized, scale_out, rows, hidden, stream); break;
    case 6: launch_warp_rmsnorm_quant<6>(input, gamma, beta, eps, quantized, scale_out, rows, hidden, stream); break;
    case 7: launch_warp_rmsnorm_quant<7>(input, gamma, beta, eps, quantized, scale_out, rows, hidden, stream); break;
    case 8: launch_warp_rmsnorm_quant<8>(input, gamma, beta, eps, quantized, scale_out, rows, hidden, stream); break;
    case 9: launch_warp_rmsnorm_quant<9>(input, gamma, beta, eps, quantized, scale_out, rows, hidden, stream); break;
    case 10: launch_warp_rmsnorm_quant<10>(input, gamma, beta, eps, quantized, scale_out, rows, hidden, stream); break;
    case 11: launch_warp_rmsnorm_quant<11>(input, gamma, beta, eps, quantized, scale_out, rows, hidden, stream); break;
    case 12: launch_warp_rmsnorm_quant<12>(input, gamma, beta, eps, quantized, scale_out, rows, hidden, stream); break;
    case 13: launch_warp_rmsnorm_quant<13>(input, gamma, beta, eps, quantized, scale_out, rows, hidden, stream); break;
    case 14: launch_warp_rmsnorm_quant<14>(input, gamma, beta, eps, quantized, scale_out, rows, hidden, stream); break;
    case 15: launch_warp_rmsnorm_quant<15>(input, gamma, beta, eps, quantized, scale_out, rows, hidden, stream); break;
    case 16: launch_warp_rmsnorm_quant<16>(input, gamma, beta, eps, quantized, scale_out, rows, hidden, stream); break;
    default:
    {
        constexpr int32_t WARPS = kRmsnormWarpsPerBlock;
        int const blocks = (rows + WARPS - 1) / WARPS;
        warp_rmsnorm_quant_stream_kernel<WARPS>
            <<<blocks, 32 * WARPS, 0, stream>>>(input, gamma, beta, eps, quantized, scale_out, rows, hidden);
        break;
    }
    }
    return cudaGetLastError();
}

cudaError_t launch_pertoken_quant(
    __half const* input, signed char* quantized, float* scale_out, int rows, int cols, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0)
    {
        return cudaSuccess;
    }
    int32_t const vpl = warp_vpl_for(cols);
    if (vpl != 0)
    {
        switch (vpl)
        {
        case 1: launch_warp_pertoken_quant<1>(input, quantized, scale_out, rows, cols, stream); break;
        case 2: launch_warp_pertoken_quant<2>(input, quantized, scale_out, rows, cols, stream); break;
        case 3: launch_warp_pertoken_quant<3>(input, quantized, scale_out, rows, cols, stream); break;
        case 4: launch_warp_pertoken_quant<4>(input, quantized, scale_out, rows, cols, stream); break;
        case 5: launch_warp_pertoken_quant<5>(input, quantized, scale_out, rows, cols, stream); break;
        case 6: launch_warp_pertoken_quant<6>(input, quantized, scale_out, rows, cols, stream); break;
        case 7: launch_warp_pertoken_quant<7>(input, quantized, scale_out, rows, cols, stream); break;
        case 8: launch_warp_pertoken_quant<8>(input, quantized, scale_out, rows, cols, stream); break;
        case 9: launch_warp_pertoken_quant<9>(input, quantized, scale_out, rows, cols, stream); break;
        case 10: launch_warp_pertoken_quant<10>(input, quantized, scale_out, rows, cols, stream); break;
        case 11: launch_warp_pertoken_quant<11>(input, quantized, scale_out, rows, cols, stream); break;
        case 12: launch_warp_pertoken_quant<12>(input, quantized, scale_out, rows, cols, stream); break;
        case 13: launch_warp_pertoken_quant<13>(input, quantized, scale_out, rows, cols, stream); break;
        case 14: launch_warp_pertoken_quant<14>(input, quantized, scale_out, rows, cols, stream); break;
        case 15: launch_warp_pertoken_quant<15>(input, quantized, scale_out, rows, cols, stream); break;
        case 16: launch_warp_pertoken_quant<16>(input, quantized, scale_out, rows, cols, stream); break;
        default: break;
        }
        return cudaGetLastError();
    }

    size_t const rowbytes = (size_t) cols * sizeof(__half);
    if (cols % 2 == 0 && rowbytes <= kMaxSmemBytes)
    {
        constexpr int32_t BLOCK = 192;
        block_pertoken_quant_h2_kernel<BLOCK>
            <<<rows, BLOCK, rowbytes, stream>>>(input, quantized, scale_out, rows, cols);
    }
    else
    {
        constexpr int32_t WARPS = kQuantWarpsPerBlock;
        int const blocks = (rows + WARPS - 1) / WARPS;
        warp_pertoken_quant_stream_kernel<WARPS>
            <<<blocks, 32 * WARPS, 0, stream>>>(input, quantized, scale_out, rows, cols);
    }
    return cudaGetLastError();
}

int32_t flatten_rows(Dims const& dims, int32_t& rows, int32_t& columns) noexcept
{
    if (dims.nbDims < 1 || dims.nbDims > Dims::MAX_DIMS)
    {
        return 0;
    }
    columns = dims.d[dims.nbDims - 1];
    if (columns <= 0)
    {
        return 0;
    }
    int64_t flat{1};
    for (int32_t index = 0; index + 1 < dims.nbDims; ++index)
    {
        if (dims.d[index] <= 0)
        {
            return 0;
        }
        flat *= dims.d[index];
    }
    if (flat <= 0 || flat > std::numeric_limits<int32_t>::max())
    {
        return 0;
    }
    rows = static_cast<int32_t>(flat);
    return 1;
}

// -------------------------------------------------------------------------------------------
// RMSNorm + dynamic per-token INT8 quantization plugin
// -------------------------------------------------------------------------------------------

class RmsnormQuantPlugin : public IPluginV2DynamicExt
{
public:
    RmsnormQuantPlugin(float eps, bool dynamicActivationScaling) noexcept
        : mEps(eps)
        , mDynActScaling(dynamicActivationScaling)
    {
    }

    RmsnormQuantPlugin(void const* data, size_t length)
    {
        if (data == nullptr || length < getSerializationSize())
        {
            throw std::runtime_error("TrackVLARmsnormQuant: bad serialized buffer");
        }
        char const* cursor = static_cast<char const*>(data);
        std::memcpy(&mEps, cursor, sizeof(float));
        cursor += sizeof(float);
        int32_t flag{0};
        std::memcpy(&flag, cursor, sizeof(int32_t));
        mDynActScaling = flag != 0;
    }

    IPluginV2DynamicExt* clone() const noexcept override
    {
        return new RmsnormQuantPlugin(mEps, mDynActScaling);
    }

    char const* getPluginType() const noexcept override
    {
        return kRmsnormQuantName;
    }

    char const* getPluginVersion() const noexcept override
    {
        return kPluginVersion;
    }

    int32_t getNbOutputs() const noexcept override
    {
        return mDynActScaling ? 2 : 1;
    }

    DimsExprs getOutputDimensions(
        int32_t index, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& builder) noexcept override
    {
        try
        {
            if (index == 0)
            {
                return inputs[0];
            }
            DimsExprs out;
            out.nbDims = inputs[0].nbDims;
            for (int32_t i = 0; i + 1 < out.nbDims; ++i)
            {
                out.d[i] = inputs[0].d[i];
            }
            out.d[out.nbDims - 1] = builder.constant(1);
            return out;
        }
        catch (std::exception const&)
        {
            return DimsExprs{};
        }
    }

    bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        int32_t const nbPluginInputs = mDynActScaling ? 4 : 4;
        if (pos < nbPluginInputs)
        {
            DataType const expected = (pos == 3) ? DataType::kFLOAT : DataType::kHALF;
            return inOut[pos].type == expected && inOut[pos].format == TensorFormat::kLINEAR;
        }
        if (pos == nbPluginInputs)
        {
            return inOut[pos].type == DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
        }
        if (pos == nbPluginInputs + 1)
        {
            return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
        }
        return false;
    }

    void configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override
    {
    }

    size_t getWorkspaceSize(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) const noexcept override
    {
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
        void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        int32_t rows = 0;
        int32_t columns = 0;
        if (flatten_rows(inputDesc[0].dims, rows, columns) == 0 || columns % 2 != 0)
        {
            return 1;
        }
        if (!mDynActScaling)
        {
            // Per-tensor scaling is not emitted by this build path; keep the contract explicit.
            return 1;
        }
        cudaError_t const status = launch_rmsnorm_quant(reinterpret_cast<__half const*>(inputs[0]),
            reinterpret_cast<__half const*>(inputs[1]), reinterpret_cast<__half const*>(inputs[2]), mEps,
            reinterpret_cast<signed char*>(outputs[0]), reinterpret_cast<float*>(outputs[1]), rows, columns,
            stream);
        return status == cudaSuccess ? 0 : 1;
    }

    DataType getOutputDataType(int32_t index, DataType const*, int32_t) const noexcept override
    {
        return index == 0 ? DataType::kINT8 : DataType::kFLOAT;
    }

    int32_t initialize() noexcept override
    {
        return 0;
    }

    void terminate() noexcept override {}

    size_t getSerializationSize() const noexcept override
    {
        return sizeof(float) + sizeof(int32_t);
    }

    void serialize(void* buffer) const noexcept override
    {
        char* cursor = static_cast<char*>(buffer);
        std::memcpy(cursor, &mEps, sizeof(float));
        cursor += sizeof(float);
        int32_t const flag = mDynActScaling ? 1 : 0;
        std::memcpy(cursor, &flag, sizeof(int32_t));
    }

    void destroy() noexcept override
    {
        delete this;
    }

    void setPluginNamespace(char const* pluginNamespace) noexcept override
    {
        mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
    }

    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    float mEps{1e-5f};
    bool mDynActScaling{true};
    std::string mNamespace{kNamespace};
};

// -------------------------------------------------------------------------------------------
// Dynamic per-token INT8 quantization plugin
// -------------------------------------------------------------------------------------------

class QuantizePerTokenPlugin : public IPluginV2DynamicExt
{
public:
    QuantizePerTokenPlugin() noexcept = default;

    QuantizePerTokenPlugin(void const* data, size_t length)
    {
        if (data == nullptr || length < getSerializationSize())
        {
            throw std::runtime_error("TrackVLAQuantizePerToken: bad serialized buffer");
        }
        int32_t flag{0};
        std::memcpy(&flag, data, sizeof(int32_t));
        (void) flag;
    }

    IPluginV2DynamicExt* clone() const noexcept override
    {
        return new QuantizePerTokenPlugin();
    }

    char const* getPluginType() const noexcept override
    {
        return kQuantizePerTokenName;
    }

    char const* getPluginVersion() const noexcept override
    {
        return kPluginVersion;
    }

    int32_t getNbOutputs() const noexcept override
    {
        return 2;
    }

    DimsExprs getOutputDimensions(
        int32_t index, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& builder) noexcept override
    {
        try
        {
            if (index == 0)
            {
                return inputs[0];
            }
            DimsExprs out;
            out.nbDims = inputs[0].nbDims;
            for (int32_t i = 0; i + 1 < out.nbDims; ++i)
            {
                out.d[i] = inputs[0].d[i];
            }
            out.d[out.nbDims - 1] = builder.constant(1);
            return out;
        }
        catch (std::exception const&)
        {
            return DimsExprs{};
        }
    }

    bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        if (pos == 0)
        {
            return inOut[pos].type == DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
        }
        if (pos == 1)
        {
            return inOut[pos].type == DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
        }
        if (pos == 2)
        {
            return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
        }
        return false;
    }

    void configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override
    {
    }

    size_t getWorkspaceSize(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) const noexcept override
    {
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
        void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        int32_t rows = 0;
        int32_t columns = 0;
        if (flatten_rows(inputDesc[0].dims, rows, columns) == 0)
        {
            return 1;
        }
        cudaError_t const status = launch_pertoken_quant(reinterpret_cast<__half const*>(inputs[0]),
            reinterpret_cast<signed char*>(outputs[0]), reinterpret_cast<float*>(outputs[1]), rows, columns, stream);
        return status == cudaSuccess ? 0 : 1;
    }

    DataType getOutputDataType(int32_t index, DataType const*, int32_t) const noexcept override
    {
        return index == 0 ? DataType::kINT8 : DataType::kFLOAT;
    }

    int32_t initialize() noexcept override
    {
        return 0;
    }

    void terminate() noexcept override {}

    size_t getSerializationSize() const noexcept override
    {
        return sizeof(int32_t);
    }

    void serialize(void* buffer) const noexcept override
    {
        int32_t const flag = 0;
        std::memcpy(buffer, &flag, sizeof(int32_t));
    }

    void destroy() noexcept override
    {
        delete this;
    }

    void setPluginNamespace(char const* pluginNamespace) noexcept override
    {
        mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
    }

    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    std::string mNamespace{kNamespace};
};

// -------------------------------------------------------------------------------------------
// Creators.  The field set mirrors the TRT-LLM plugin names so the build-time emitter patch
// can keep using the same call shape.
// -------------------------------------------------------------------------------------------

class RmsnormQuantCreator : public IPluginCreator
{
public:
    RmsnormQuantCreator()
    {
        mAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
        mAttributes.emplace_back(PluginField("dyn_act_scaling", nullptr, PluginFieldType::kINT32, 1));
        mFields.nbFields = static_cast<int32_t>(mAttributes.size());
        mFields.fields = mAttributes.data();
    }

    char const* getPluginName() const noexcept override
    {
        return kRmsnormQuantName;
    }

    char const* getPluginVersion() const noexcept override
    {
        return kPluginVersion;
    }

    PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mFields;
    }

    IPluginV2* createPlugin(char const* name, PluginFieldCollection const* fields) noexcept override
    {
        float eps = 1e-5f;
        bool dynamicActivationScaling = true;
        for (int32_t i = 0; fields != nullptr && i < fields->nbFields; ++i)
        {
            PluginField const& field = fields->fields[i];
            if (field.data == nullptr)
            {
                continue;
            }
            if (std::strcmp(field.name, "eps") == 0 && field.length == 1)
            {
                std::memcpy(&eps, field.data, sizeof(float));
            }
            else if (std::strcmp(field.name, "dyn_act_scaling") == 0 && field.length == 1)
            {
                int32_t value{0};
                std::memcpy(&value, field.data, sizeof(int32_t));
                dynamicActivationScaling = value != 0;
            }
        }
        auto* plugin = new RmsnormQuantPlugin(eps, dynamicActivationScaling);
        plugin->setPluginNamespace(mNamespace.c_str());
        (void) name;
        return plugin;
    }

    IPluginV2* deserializePlugin(char const* name, void const* data, size_t length) noexcept override
    {
        (void) name;
        return new RmsnormQuantPlugin(data, length);
    }

    void setPluginNamespace(char const* pluginNamespace) noexcept override
    {
        mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
    }

    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    std::string mNamespace{kNamespace};
    std::vector<PluginField> mAttributes;
    PluginFieldCollection mFields{};
};

class QuantizePerTokenCreator : public IPluginCreator
{
public:
    QuantizePerTokenCreator()
    {
        mFields.nbFields = 0;
        mFields.fields = nullptr;
    }

    char const* getPluginName() const noexcept override
    {
        return kQuantizePerTokenName;
    }

    char const* getPluginVersion() const noexcept override
    {
        return kPluginVersion;
    }

    PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mFields;
    }

    IPluginV2* createPlugin(char const* name, PluginFieldCollection const*) noexcept override
    {
        auto* plugin = new QuantizePerTokenPlugin();
        plugin->setPluginNamespace(mNamespace.c_str());
        (void) name;
        return plugin;
    }

    IPluginV2* deserializePlugin(char const* name, void const* data, size_t length) noexcept override
    {
        (void) name;
        return new QuantizePerTokenPlugin(data, length);
    }

    void setPluginNamespace(char const* pluginNamespace) noexcept override
    {
        mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
    }

    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    std::string mNamespace{kNamespace};
    PluginFieldCollection mFields{};
};

RmsnormQuantCreator& rmsnormQuantCreator()
{
    static RmsnormQuantCreator creator;
    return creator;
}

QuantizePerTokenCreator& quantizePerTokenCreator()
{
    static QuantizePerTokenCreator creator;
    return creator;
}

// Register at dlopen time so that both the builder and the deserialiser find the creators.
struct PluginRegistrar
{
    PluginRegistrar()
    {
        IPluginRegistry* registry = getPluginRegistry();
        if (registry != nullptr)
        {
            registry->registerCreator(rmsnormQuantCreator(), kNamespace);
            registry->registerCreator(quantizePerTokenCreator(), kNamespace);
        }
    }
};

PluginRegistrar const gPluginRegistrar{};

// Loading contract: the creators above are registered from the static initialiser
// ``gPluginRegistrar``, which runs when the shared library is dlopen-ed. Load the library
// with ``ctypes.CDLL(path)`` before building or deserialising an engine that uses it; the
// library deliberately does not export the TRT 10 ``getCreators`` entry point so a
// subsequent ``IPluginRegistry::load_library`` cannot re-register the same creators under a
// second namespace.

} // namespace trackvla_llm
