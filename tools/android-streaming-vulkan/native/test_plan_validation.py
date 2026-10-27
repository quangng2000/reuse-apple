#!/usr/bin/env python3
"""Exercise native, pre-allocation rejection of malformed model plans."""
import struct
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = Path(os.environ.get("REUSE_VULKAN_TEST_CLI", ROOT / "generated/build-host/reuse-vulkan-cli"))
PLAN = Path(os.environ.get("REUSE_VULKAN_TEST_PLAN", ROOT / "generated/plan.bin"))


class NativePlanValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = PLAN.read_bytes()
        cls.header = struct.unpack_from("<6I", cls.original, 8)
        _, cls.arena, cls.ops, cls.inputs, cls.outputs, cls.states = cls.header
        cls.state_start = 32 + (cls.inputs + cls.outputs) * 8
        cls.op_start = cls.state_start + cls.states * 12
        cls.opcodes = [struct.unpack_from("<I", cls.original, cls.op_start + i * 132)[0] for i in range(cls.ops)]

    def execute(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.bin"
            path.write_bytes(data)
            return subprocess.run([str(CLI), "validate-plan", str(path)], capture_output=True, text=True, timeout=10)

    def mutated(self, offset, value):
        data = bytearray(self.original)
        struct.pack_into("<I", data, offset, value)
        return data

    def reject(self, data):
        result = self.execute(data)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("Vulkan error:", result.stderr)

    def op_word(self, opcode, word):
        return self.op_start + self.opcodes.index(opcode) * 132 + word * 4

    def test_valid_plan(self):
        result = self.execute(self.original)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"valid":true', result.stdout)

    def test_truncated_and_trailing_bytes(self):
        for data in (self.original[:31], self.original[:-1], self.original + b"\0"):
            with self.subTest(length=len(data)):
                self.reject(data)

    def test_magic_and_version(self):
        self.reject(b"INVALID!" + self.original[8:])
        self.reject(self.mutated(8, 999))

    def test_excessive_arena_and_record_counts(self):
        for offset in (12, 16, 20, 24, 28):
            with self.subTest(offset=offset):
                self.reject(self.mutated(offset, 0xFFFFFFFF))

    def test_spectrum_size_and_bounds(self):
        self.reject(self.mutated(36, 322))
        self.reject(self.mutated(32, self.arena))

    def test_overlapping_input_ports(self):
        first_offset = struct.unpack_from("<I", self.original, 32)[0]
        self.reject(self.mutated(40, first_offset))

    def test_bad_state_mapping(self):
        self.reject(self.mutated(self.state_start, 64))
        self.reject(self.mutated(self.state_start + 8, 1))

    def test_unknown_opcode(self):
        self.reject(self.mutated(self.op_start, 0))
        self.reject(self.mutated(self.op_start, 107))

    def test_wrong_address_masks(self):
        self.reject(self.mutated(self.op_start + 128, 2))
        self.reject(self.mutated(self.op_start + 128, 0xFFFFFFFF))

    def test_output_extent_overrun(self):
        self.reject(self.mutated(self.op_start + 4, self.arena - 1))

    def test_writing_committed_state(self):
        state_input = struct.unpack_from("<I", self.original, self.state_start)[0]
        self.reject(self.mutated(self.op_start + 4, state_input))

    def test_zero_matrix_shape(self):
        self.reject(self.mutated(self.op_word(7, 6), 0))

    def test_matrix_stride_overflow(self):
        data = self.mutated(self.op_word(7, 9), 2)
        struct.pack_into("<I", data, self.op_word(7, 10), 0xFFFFFFFF)
        self.reject(data)

    def test_invalid_convolution_groups_and_stride(self):
        self.reject(self.mutated(self.op_word(8, 22), 0))
        self.reject(self.mutated(self.op_word(8, 16), 0))

    def test_invalid_specialized_convolution_admission(self):
        self.reject(self.mutated(self.op_word(8, 27), 2))
        for i, opcode in enumerate(self.opcodes):
            offset = self.op_start + i * 132
            if opcode == 8 and struct.unpack_from('<I', self.original, offset + 27 * 4)[0] == 1:
                for word, value in [(14, 1), (15, 1), (17, 2), (19, 0), (20, 0xffffffff), (25, 1), (26, 1)]:
                    with self.subTest(word=word):
                        self.reject(self.mutated(offset + word * 4, value))
                return
        self.skipTest('Historical plan has no specialized convolution hint')

    def test_invalid_layernorm_epsilon(self):
        self.reject(self.mutated(self.op_word(9, 8), 0xBF800000))
        self.reject(self.mutated(self.op_word(9, 8), 0x7FC00000))

    def test_invalid_recurrent_width(self):
        self.reject(self.mutated(self.op_word(11, 14), 32))

    def test_invalid_unary_operation(self):
        self.reject(self.mutated(self.op_word(1, 6), 99))

    def test_invalid_strided_view_shape(self):
        self.reject(self.mutated(self.op_word(4, 8), 0))

    def concat_record(self, cache):
        for i, opcode in enumerate(self.opcodes):
            offset = self.op_start + i * 132
            if opcode == 13 and bool(struct.unpack_from('<I', self.original, offset + 40)[0]) == cache:
                return offset
        self.skipTest('This historical plan has no matching fused concat record')

    def test_concat_invalid_shape_flags_and_reserved_fields(self):
        offset = self.concat_record(True)
        for word, value in [(6, 0), (7, 0), (8, 0), (6, 0xffffffff), (10, 2), (11, 1), (31, 1)]:
            with self.subTest(word=word):
                self.reject(self.mutated(offset + word * 4, value))

    def test_concat_source_and_output_bounds_and_aliases(self):
        offset = self.concat_record(True)
        output = struct.unpack_from('<I', self.original, offset + 4)[0]
        for word in [3, 4, 5]:
            self.reject(self.mutated(offset + word * 4, self.arena - 1))
            self.reject(self.mutated(offset + word * 4, output))

    def test_concat_cache_crop_and_exact_state_mapping(self):
        offset = self.concat_record(True)
        a, b = struct.unpack_from('<2I', self.original, offset + 24)
        for drop in [0, a + b, a + b + 1, 0xffffffff]:
            self.reject(self.mutated(offset + 36, drop))
        cache = struct.unpack_from('<I', self.original, offset + 20)[0]
        self.reject(self.mutated(offset + 20, cache + 1))
        prior = struct.unpack_from('<I', self.original, offset + 12)[0]
        self.reject(self.mutated(offset + 12, prior + 1))
        state_input = struct.unpack_from('<I', self.original, self.state_start)[0]
        self.reject(self.mutated(offset + 20, state_input))

    def test_concat_address_mask_tracks_secondary_write(self):
        offset = self.concat_record(True)
        self.reject(self.mutated(offset + 128, 0x1a))
        offset = self.concat_record(False)
        self.reject(self.mutated(offset + 128, 0x3a))
        self.reject(self.mutated(offset + 20, 64))
        self.reject(self.mutated(offset + 36, 1))


if __name__ == "__main__":
    unittest.main()
