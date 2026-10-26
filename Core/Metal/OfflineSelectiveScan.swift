// Independent adaptation of the existing offline Mamba recurrence.
// SPDX-License-Identifier: OpenMDW-1.1
import MLX

enum OfflineScanExecution { case simd16, serial }

/// FP32 sequence scan with shared delta/gate preparation, no chunking.
final class OfflineSelectiveScan {
    private let prepare = MLXFast.metalKernel(
        name: "reuse_offline_prepare_step_gate", inputNames: ["delta", "z", "bias"],
        outputNames: ["step", "gate"], source: OfflineScanSource.preparation)
    private let scan: MLXFast.MLXFastKernel
    private let lanesPerChannel: Int

    init(execution: OfflineScanExecution = .simd16) {
        lanesPerChannel = execution == .simd16 ? 16 : 1
        scan = MLXFast.metalKernel(name: "reuse_offline_sequence_lanes\(lanesPerChannel)",
            inputNames: ["signal", "step", "dynamics", "inputCoupling", "outputCoupling", "skip", "gate"],
            outputNames: ["result"], source: execution == .simd16
                ? OfflineScanSource.recurrence : OfflineScanSource.serialRecurrence)
    }

    /// Uses the original Core scan input order: u, delta, A, B, C, D, z, bias.
    func callAsFunction(_ inputs: [MLXArray]) -> MLXArray {
        validate(inputs)
        let signal = inputs[0]
        let prepared = prepare([inputs[1], inputs[6], inputs[7]],
            grid: (signal.size, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [signal.shape, signal.shape], outputDTypes: [.float32, .float32])
        return scan([signal, prepared[0], inputs[2], inputs[3], inputs[4], inputs[5], prepared[1]],
            grid: (256 * lanesPerChannel, signal.dim(0), 1), threadGroup: (128, 1, 1),
            outputShapes: [signal.shape], outputDTypes: [.float32])[0]
    }

    private func validate(_ inputs: [MLXArray]) {
        precondition(inputs.count == 8 && inputs.allSatisfy { $0.dtype == .float32 })
        let signal = inputs[0]
        precondition(signal.ndim == 3 && signal.dim(0) > 0 && signal.dim(1) > 0 && signal.dim(2) == 256)
        precondition(inputs[1].shape == signal.shape && inputs[6].shape == signal.shape)
        precondition(inputs[2].shape == [256, 16] && inputs[5].shape == [256] && inputs[7].shape == [256])
        let couplingShape = [signal.dim(0), signal.dim(1), 16]
        precondition(inputs[3].shape == couplingShape && inputs[4].shape == couplingShape)
    }
}
