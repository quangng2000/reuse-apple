"""Fetch hash-pinned public research checkpoint/sources. No credentials required."""
import hashlib
import urllib.request
from urllib.parse import urlsplit
from common import SOURCE, GENERATED, DEFAULT_CHECKPOINT


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        if urlsplit(newurl).scheme != 'https':
            raise ValueError('Research asset redirects must use HTTPS')
        return super().redirect_request(request, fp, code, message, headers, newurl)


def fetch(url, path, sha256, expected_bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size != expected_bytes or hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
            raise ValueError(f"Integrity failure for{path.name}")
        return
    temporary = path.with_suffix(path.suffix + ".download")
    try:
        if urlsplit(url).scheme != 'https':
            raise ValueError('Research asset downloads must use HTTPS')
        opener = urllib.request.build_opener(HTTPSRedirect())
        count = 0
        with opener.open(url, timeout=120) as response, temporary.open("wb") as output:
            while data := response.read(128 * 1024):
                count += len(data)
                if count > expected_bytes:
                    raise ValueError('Research asset exceeds its pinned size')
                output.write(data)
        if count != expected_bytes or hashlib.sha256(temporary.read_bytes()).hexdigest() != sha256:
            raise ValueError(f"Integrity failure for{path.name}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    base = f"https://huggingface.co/{SOURCE['repo']}/resolve/{SOURCE['revision']}/"
    fetch(base + SOURCE['checkpoint']['upstream_path'], DEFAULT_CHECKPOINT,
          SOURCE['checkpoint']['sha256'], SOURCE['checkpoint']['bytes'])
    for item in SOURCE['reference_sources']:
        fetch(base + item['upstream_path'], GENERATED / 'upstream' / item['upstream_path'], item['sha256'], item['bytes'])
    print("Verified NSCLv1 research-only checkpoint and pinned reference sources")


if __name__ == '__main__':
    main()
