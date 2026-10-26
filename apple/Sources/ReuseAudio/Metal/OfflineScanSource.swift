// Independent SIMD implementation of apple/Sources/ReuseAudio/SelectiveScan.swift's Mamba recurrence.
// Mamba reference: Copyright (c) 2023 Tri Dao, Albert Gu (Apache-2.0).
// SPDX-License-Identifier: OpenMDW-1.1

enum OfflineScanSource {
    static let preparation = """
        uint index = thread_position_in_grid.x;
        uint width = delta_shape[2];
        if (index >= delta_shape[0] * delta_shape[1] * width) return;
        float value = delta[index] + bias[index % width];
        step[index] = max(value, 0.0f) + log1p(exp(-abs(value)));
        float activation = z[index];
        gate[index] = activation / (1.0f + exp(-activation));
        """

    static let recurrence = """
        uint component = thread_position_in_grid.x % 16;
        uint channel = thread_position_in_grid.x / 16;
        uint batch = thread_position_in_grid.y;
        uint width = signal_shape[2], length = signal_shape[1];
        float coefficient = dynamics[channel * 16 + component];
        float state = 0.0f;
        for (uint position = 0; position < length; ++position) {
            uint sample = (batch * length + position) * width + channel;
            uint coupling = (batch * length + position) * 16 + component;
            float delta = step[sample], input = signal[sample];
            state = exp(delta * coefficient) * state
                  + delta * inputCoupling[coupling] * input;
            float contribution = state * outputCoupling[coupling];
            // XOR distances below 16 cannot cross either half of a SIMD32 group.
            contribution += simd_shuffle_xor(contribution, 1);
            contribution += simd_shuffle_xor(contribution, 2);
            contribution += simd_shuffle_xor(contribution, 4);
            contribution += simd_shuffle_xor(contribution, 8);
            if (component == 0)
                result[sample] = (contribution + skip[channel] * input) * gate[sample];
        }
        """

    static let serialRecurrence = """
        uint channel = thread_position_in_grid.x;
        uint batch = thread_position_in_grid.y;
        uint width = signal_shape[2], length = signal_shape[1];
        float state[16], coefficient[16];
        for (uint component = 0; component < 16; ++component) {
            state[component] = 0.0f;
            coefficient[component] = dynamics[channel * 16 + component];
        }
        for (uint position = 0; position < length; ++position) {
            uint sample = (batch * length + position) * width + channel;
            uint coupling = (batch * length + position) * 16;
            float delta = step[sample], input = signal[sample], output = 0.0f;
            for (uint component = 0; component < 16; ++component) {
                state[component] = exp(delta * coefficient[component]) * state[component]
                                 + delta * inputCoupling[coupling + component] * input;
                output += state[component] * outputCoupling[coupling + component];
            }
            result[sample] = (output + skip[channel] * input) * gate[sample];
        }
        """
}
