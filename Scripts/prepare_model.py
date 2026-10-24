"""Download and verify the pinned NVIDIA checkpoint and its original notices."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REVISION = '31f82f2efad1095034492f236e8dac9239ba713b'
REPO = 'nvidia/NVIDIA-NemotronLabs-RE-USE'
FILES = (
    ('model.safetensors', 'model.safetensors',
     '06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1'),
    ('LICENSE', 'MODEL-LICENSE.txt',
     'a3d882953f026d3e2fdd57842d9c522c8320d4c0388af2278bcf4d438d255a2c'),
    ('README.md', 'MODEL-CARD.md',
     '6fa6e175b953fdb596d5d1c10feed5ac249f249f65e40c6d9ad981650924a9f2'),
)


def verify(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'Checksum mismatch for {path.name}; expected {expected}, got {actual}')


def prepare_file(entry, resources):
    source, name, expected = entry
    path = resources / name
    if not path.exists():
        temporary = path.with_suffix(path.suffix + '.download')
        subprocess.run(['curl', '--fail', '--location', '--proto', '=https',
                        '--proto-redir', '=https', '--retry', '3', '--connect-timeout', '15',
                        '--max-time', '300', f'https://huggingface.co/{REPO}/resolve/{REVISION}/{source}',
                        '--output', str(temporary)], check=True)
        verify(temporary, expected)
        temporary.replace(path)
    verify(path, expected)


def main():
    resources = ROOT / 'Resources'
    resources.mkdir(exist_ok=True)
    for entry in FILES:
        prepare_file(entry, resources)
    provenance = dict(repo=REPO, revision=REVISION, sha256=FILES[0][2], parameters=9607747,
                      weight_dtype='float32', license='OpenMDW-1.1', sample_rate=16000,
                      port='Swift / MLX 0.29.1 / Metal; offline 4-second windows, 0.5-second overlap')
    (resources / 'ModelProvenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Verified pinned NVIDIA checkpoint and original model notices.')


if __name__ == '__main__':
    main()
