"""Inspect version bindings for the pinned 18,000-episode public dataset.

This is a provenance lookup, not a simulator-state loader. A located commit
does not certify its dependencies or make an episode replayable.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re


def load_catalog():
    root = Path(__file__).parent
    manifest = json.loads((root / 'catalog_manifest.json').read_text())
    raw = (root / 'catalog.json.gz').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['catalog_sha256']:
        raise ValueError('Episode provenance catalog checksum mismatch')
    catalog = json.loads(gzip.decompress(raw))
    if len(catalog) != manifest['episodes']:
        raise ValueError('Episode provenance catalog length mismatch')
    return manifest, catalog


def describe(episode):
    if not re.fullmatch(r'e\d{6}', episode):
        raise ValueError('Use the public episode identifier, e.g. e015401')
    manifest, rows = load_catalog()
    index = int(episode[1:])
    if index >= len(rows) or rows[index]['episode_index'] != index:
        raise ValueError(f'Episode is not in the pinned public package: {episode}')
    return dict(dataset=manifest['dataset'], dataset_revision=manifest['dataset_revision'],
                **rows[index])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', help='Public identifier, e.g. e015401; omit for summary')
    args = parser.parse_args(argv)
    try:
        result = describe(args.episode) if args.episode else load_catalog()[0]
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
