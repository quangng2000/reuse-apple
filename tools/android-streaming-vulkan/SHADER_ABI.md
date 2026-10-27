# Fixed FP32 compute shader ABI

These shaders execute the exported eight-layer graph directly through Vulkan.
They neither load ONNX Runtime nor delegate arithmetic to a CPU execution
provider. The normal Android streaming SDK remains available separately.

`build_shaders.py` compiles Vulkan 1.1 SPIR-V with the installed NDK `glslc`,
validates every module with `spirv-val`, rejects non-FP32 float types,
`RelaxedPrecision`, and fast-math decorations, and records source/module hashes.
The ordinary shaders preserve separate multiply/add rounding. `--math-variant
fma` substitutes explicit FP32 fused multiply-add in convolution and matrix
multiplication; it does not use FP16, quantization, or implicit fast math.

`--gpu-variant adreno64` optionally replaces normalization with a 64-thread
implementation. Specialization constant ID 0 receives the actual device
subgroup size from the native runtime. On a subgroup64 device, the model's
64-channel rows use subgroup shuffles with the same binary reduction tree and
no workgroup barriers. Other subgroup sizes use a shared-memory fallback.
The default `baseline` build is unchanged. `adreno64-fastindex` additionally
selects experimental integer address calculations that avoid generic division
for power-of-two dimensions; it requires its own performance validation.

The C++ runtime must check compute subgroup support, basic/shuffle/shuffle-relative
operations, and a subgroup size at least sixteen and divisible by sixteen. A
subgroup divides into independent groups of sixteen recurrent-state components.
All model coefficients and persistent states reside in the FP32 arena.

## Common descriptor

Binding 0 is a `std430` storage buffer containing `float data[]`. Each dispatch
has 128 push-constant bytes: 32 little-endian unsigned 32-bit words. Words 0–5
are opcode, output float offset, output element count, and input offsets 0–2.
Signed slice strides are represented by their two's-complement bits. Float
constants are represented by their IEEE-754 bits. Addresses count floats, not
bytes. The exporter separately records an address-field mask for state-bank
relocation; it is not a shader push constant.

## Operators

1. `unary`: word 6 selects Neg=1, Exp=2, Sigmoid=3, Softplus=4, Atan=5,
   Sqrt=6, Reciprocal=7, Tanh=8, Relu=9, Log=10, Abs=11, Sin=12, Cos=13,
   SiLU=14. Input/output are contiguous. SiLU retains the rounded sigmoid
   division followed by a separate multiplication.
2. `binary`: word 6 selects Add=1, Mul=2, Div=3, PRelu=4, Less=5, Greater=6,
   Equal=7, Sub=8, Pow=9, Max=10, Min=11. Words 8–13 hold output dimensions,
   padded to six dimensions with leading ones. Words 14–19 and 20–25 hold
   input strides; a broadcast dimension has stride zero. Boolean results use
   exact FP32 zero/one values.
3. `where`: dimensions 8–13; condition/true/false strides 14–19, 20–25,
   26–31. A nonzero condition selects true.
4. `gather`: dimensions 8–13 and signed source strides 14–19. The source
   offset already includes slice starts. This covers transpose and slice.
5. `pad`: output dimensions 8–13, input dimensions 14–19, leading padding
   20–25, fill-value bits in word 6. Only constant padding is admitted.
6. `scatter`: words 8–13 describe input dimensions, 14–19 destination
   strides. Word 2 is the input count; output offset includes the concat
   part's start. Different concat dispatches write disjoint regions.
7. `matmul`: words 6–13 are M, N, K, batch count, A batch stride, B batch
   stride, transpose-A, transpose-B. Zero batch stride broadcasts an input.
   Optional following Add uses word 14: 0=none, 1=column bias, 2=full output
   tensor, with the addend in input offset word 5. Word 15=1 adds Softplus
   after dot and bias. These remain separate FP32 arithmetic boundaries.
8. `conv`: words 6–24 are output N/C/H/W, input N/C/H/W, kernel H/W,
   stride H/W, leading padding H/W, dilation H/W, groups, bias-present,
   transpose-convolution. Word 25=1 changes input/output to NHWC; word
   26=1 applies SiLU after the optional bias. Weights retain ONNX's exact
   FP32 layout. Conv1D is normalized to H=1.
9. `layernorm`: words 6–10 are rows, width, epsilon bits, scale-present,
   bias-present. Input offsets 1/2 hold scale/bias. Word 11=0 means the
   contiguous last-axis layout. Word 11=1 means channel normalization in
   NCHW, with spatial H×W in word 12. Optional PRelu uses flag word 13,
   slope offset word 14, and slope count word 15 (one or width). The
   normalization and PRelu remain separately rounded FP32 operations.
10. `frequency`: output offset word 1 is the output values; old state,
    input U, delta, B, C, gate, A, D occupy words 3–10. New state offset
    is word 11; bins/channels/components are words 12–14. Components must
    be sixteen. Each group of sixteen GPU lanes traverses frequency in
    order and reduces its sixteen state products using a balanced tree.
11. `temporal`: same inputs as frequency; old/new states have
    `[bins, channels, 16]` layout. One update per bin/channel is performed.

## Dispatch dimensions

The ordinary elementwise/layout/convolution shaders use 128 threads and
`ceil(word2/128)` groups. Matrix multiplication uses 16×8 threads, with group
counts `ceil(N/16), ceil(M/8), batch`. Layer normalization uses 256 threads
and one group per row. Frequency uses 64 threads and `channels×16/64` groups;
temporal uses 64 threads and `ceil(values×16/64)` groups.

The optional `matmul_tiled` shader computes a 16-row × 32-column tile with
four accumulators per thread: dispatch `ceil(N/32), ceil(M/16), batch`.
`conv_tiled` uses the same tile shape with output channels as rows and H×W
as columns: dispatch `ceil(Hout×Wout/32), ceil(Cout/16), N`. The latter is
only valid for group-one forward NCHW convolution without fused SiLU.
Kernel selection and dispatch dimensions must agree.

`finite_check` uses binding 1 for one atomic status word. Its push constants
are only offset/count in the first two words; it scans the exponent bits of
every output/state and sets the flag for either infinity or NaN. The runtime
reads four bytes, rejects nonfinite output, and commits the state bank only
after a successful frame. It does not copy all state tensors to the CPU.

## Numerical checks

`test_gpu_kernels.py` executes miniature plans on the actual Vulkan engine and
checks strided/broadcast operations, odd-sized matrix multiplication, grouped
and transposed convolution, fused activations, normalization, and both recurrent
updates. Full graph spectra/all-state parity and public SDK waveform/audio
gates are separate requirements. Host GPU timing is a development diagnostic;
only the physical Android-phone benchmark measures the requested device.

The Exp polynomial comes from ONNX Runtime 1.23.2 ARM64 MLAS, retaining its
explicit FP32 fused operations and constants. Its MIT notice is included at
`shaders/MLAS-MIT.txt`. The model adaptation retains the existing streaming
model's NVIDIA noncommercial research/education license.

### Opcode 13: fused concat with optional cache suffix

`concat_cache` uses the standard 128-thread linear dispatch. `p1` is the full
concatenated output, `p2` its element count, `p3`/`p4` the two input pieces,
`p5` the optional next-state cache, `p6`/`p7` the per-row element counts of
those pieces, and `p8` the outer row count. `p9` is the number of elements to
drop from each output row when `p10=1` enables the cache write. Unused words
11–31 are zero. The address mask includes 1, 3 and 4, plus 5 only when caching.
For caching, the admitted plan requires `0 < p9 < p6+p7`; the cache output is
the exact paired next-state port for `p3`, with `p8*(p6+p7-p9)` elements.
The original state count equals `p8*p6`, and `p9==p7` preserves that count. All written
spans are disjoint from each other and both input spans. A single loaded FP32
value is copied to the concat output and, when applicable, its cache suffix;
no FP32 arithmetic occurs, including for negative zero.

### Optional dense 3×3 convolution variants

`--math-variant fma --gpu-variant adreno64-conv36` or `adreno64-conv72`
retains the Adreno64 layer-normalization variant and the existing 16×8
convolution dispatch. For admitted dense NCHW 3×3 convolutions with one output
time row, horizontal stride/dilation 1, horizontal padding 1, and input time
extent `1+2*dilationH`, the shader uses constant kernel indexing and 36 or 72
reduction elements per shared tile. Ordered `ci,ky,kx` FP32 FMA accumulation
is unchanged. Other admitted tiled convolution shapes use the original
16-element fallback. These variants require explicit FMA mode; they do not
use FP16, relaxed precision, fast math, or changed model weights.

The selected separate-pipeline variant is
`--math-variant fma --gpu-variant adreno64-conv72-separated`. Its dedicated
`conv_3x3_tiled` kernel has 13,824 bytes of shared storage. The ordinary
`conv_tiled` remains byte-identical to the previous FMA kernel, retaining its
3,072-byte shared footprint for all other shapes. An explicit `p27=1` on an
eligible convolution descriptor requests the dedicated pipeline when tiled
convolution is enabled; file presence alone never selects it. The host
validates the complete dense3×3 predicate before accepting this flag. Legacy
plans with `p27=0` use their previous pipeline even if the new shader is present.
