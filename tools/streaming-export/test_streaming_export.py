"""Fast topology/integrity tests, no inference downloads or model mutation."""
import tempfile
import unittest
from pathlib import Path
from common import selected_weights, DEFAULT_CHECKPOINT
from export_streaming import StreamingGraph


class StreamingExportTests(unittest.TestCase):
    def test_rejects_unsupported_depth_before_loading(self):
        for depth in [0,2,13,30]:
            with self.assertRaisesRegex(ValueError,'3..12'):
                selected_weights(Path('/missing'), depth)

    def test_rejects_different_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.safetensors'; p.write_bytes(b'different model')
            with self.assertRaisesRegex(ValueError,'SHA256'):
                selected_weights(p,4)

    def test_all_advertised_depths_have_complete_state_topology(self):
        for depth in range(3,13):
            _, weights=selected_weights(DEFAULT_CHECKPOINT,depth)
            graph=StreamingGraph(weights,depth)
            model=graph.build()
            self.assertEqual(15+2*depth,len(graph.states))
            self.assertEqual(len(graph.states),len({s['key'] for s in graph.states}))
            self.assertEqual([1,2,2,323],graph.states[0]['shape'])
            self.assertEqual(2*depth,sum(n.op_type=='Scan' for n in model.graph.node))
            self.assertEqual(2+len(graph.states),len(model.graph.input))
            self.assertEqual(2+len(graph.states),len(model.graph.output))
            for item in graph.states:
                self.assertEqual('float32',item['dtype'])
                self.assertTrue(all(x>0 for x in item['shape']))


if __name__=='__main__':unittest.main()
