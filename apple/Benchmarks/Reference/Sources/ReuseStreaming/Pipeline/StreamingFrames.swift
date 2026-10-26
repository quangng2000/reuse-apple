// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import Foundation

/// Bounded input buffer. Frames become available only after their samples arrive.
/// Reflection at the beginning waits for 40 ms; end reflection occurs only at EOF.
struct StreamingFrames {
    private var buffer = [Float]()
    private var recent = [Float]()
    private var initialized = false
    private var total = 0
    private var emitted = 0

    mutating func push(_ samples: [Float]) -> [[Float]] {
        total += samples.count
        recent = Array((recent + samples).suffix(641))
        buffer += samples
        if !initialized && buffer.count >= 640 {
            buffer = Array(buffer[1...320].reversed()) + buffer
            initialized = true
        }
        return initialized ? drain() : []
    }

    mutating func finish() -> [[Float]] {
        let target = 1 + (total + 319) / 320
        buffer += Array(recent.dropLast().reversed())
        return drain(limit: target - emitted)
    }

    private mutating func drain(limit: Int = Int.max) -> [[Float]] {
        var frames = [[Float]]()
        while buffer.count >= 640 && frames.count < limit {
            frames.append(Array(buffer.prefix(640)))
            buffer.removeFirst(320)
            emitted += 1
        }
        return frames
    }
}
