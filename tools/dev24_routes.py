"""Resolve and verify the versioned dev24 routes used by evaluation code."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / 'evaluation/routes/dev24_v1/selection_manifest.json'


def resolve_routes(manifest_path=None, shard=None):
    manifest_path = Path(manifest_path or DEFAULT_MANIFEST).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest['version'] != 'dev24_v1':
        raise ValueError('This entry point requires dev24_v1')
    if shard is not None and shard not in (1, 2, 3, 4):
        raise ValueError('shard must be 1, 2, 3, or 4')
    spec = manifest['xml'] if shard is None else manifest['shards'][shard - 1]
    path = manifest_path.parent / spec['file']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec['sha256']:
        raise ValueError('Route XML differs from the frozen manifest: ' + str(path))
    ids = [r.get('id') for r in ET.parse(path).getroot()]
    expected = 24 if shard is None else 6
    if ids != spec['route_ids_in_xml_order'] or len(ids) != expected or len(set(ids)) != expected:
        raise ValueError('Invalid dev24 route IDs/count/order')
    return dict(subset=manifest['version'], route_manifest=str(manifest_path),
                routes=str(path), route_count=len(ids), route_ids=ids,
                routes_sha256=digest, shard=shard)
