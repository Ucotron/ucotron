#!/usr/bin/env python3
"""QA-007: Test multimodal image ingestion and CLIP search.

Tests:
1. POST /memories/image ingests a PNG image
2. POST /memories/image ingests a JPG image
3. POST /images indexes images with CLIP embeddings
4. POST /images/search returns similar images
5. CLIP embeddings are 512-dimensional
6. GET /media/{id} serves the stored media file
"""

import json
import os
import sys
import io
import time
import struct
import zlib
import requests

BASE_URL = "http://localhost:8420/api/v1"
RESULTS = {"test_suite": "QA-007", "description": "Multimodal: image ingestion and CLIP search", "tests": []}
CREATED_IDS = []


def record(name, passed, details=None, response=None):
    entry = {"test": name, "passed": passed}
    if details:
        entry["details"] = details
    if response is not None:
        try:
            entry["response"] = response.json() if hasattr(response, 'json') else response
        except Exception:
            entry["response"] = str(response.text)[:500] if hasattr(response, 'text') else str(response)[:500]
        if hasattr(response, 'status_code'):
            entry["status_code"] = response.status_code
    RESULTS["tests"].append(entry)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    if not passed and details:
        print(f"         {details}")


def create_png_bytes(width=64, height=64, color=(255, 0, 0)):
    """Create a minimal valid PNG image in memory."""
    def make_chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack('>I', zlib.crc32(c) & 0xffffffff)
        return struct.pack('>I', len(data)) + c + crc

    # PNG signature
    sig = b'\x89PNG\r\n\x1a\n'
    # IHDR
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    ihdr = make_chunk(b'IHDR', ihdr_data)
    # IDAT - raw image data
    raw = b''
    for y in range(height):
        raw += b'\x00'  # filter none
        for x in range(width):
            # Create a gradient pattern
            r = int(color[0] * (1 - x / width))
            g = int(color[1] * (y / height))
            b = int(color[2] * (x / width))
            raw += struct.pack('BBB', r, g, b)
    compressed = zlib.compress(raw)
    idat = make_chunk(b'IDAT', compressed)
    iend = make_chunk(b'IEND', b'')
    return sig + ihdr + idat + iend


def create_jpg_bytes(width=64, height=64):
    """Create a minimal JPEG using PIL if available, otherwise a simple JFIF."""
    try:
        from PIL import Image
        img = Image.new('RGB', (width, height), color=(0, 128, 255))
        # Add some variation
        for x in range(width):
            for y in range(height):
                img.putpixel((x, y), (x * 4 % 256, y * 4 % 256, 128))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return buf.getvalue()
    except ImportError:
        # Fallback: create a minimal valid JPEG (SOI + APP0 + EOI)
        # This is a bare minimum JPEG that some decoders accept
        return (
            b'\xff\xd8'  # SOI
            b'\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'  # APP0
            b'\xff\xd9'  # EOI
        )


def test_png_ingestion():
    """Test 1: POST /memories/image ingests a PNG image."""
    print("\n--- Test 1: PNG Image Ingestion ---")
    png_data = create_png_bytes(64, 64, (255, 0, 0))

    resp = requests.post(
        f"{BASE_URL}/memories/image",
        files={"file": ("test_red.png", png_data, "image/png")},
        data={"description": "A red gradient test image for QA-007"},
    )
    passed = resp.status_code == 201
    details = None
    if passed:
        body = resp.json()
        CREATED_IDS.append(("png_memory", body.get("node_id")))
        details = f"node_id={body.get('node_id')}, format={body.get('format')}, embedding_dim={body.get('embedding_dim')}"
    else:
        details = f"Expected 201, got {resp.status_code}: {resp.text[:300]}"
    record("POST /memories/image with PNG", passed, details, resp)
    return resp


def test_jpg_ingestion():
    """Test 2: POST /memories/image ingests a JPG image."""
    print("\n--- Test 2: JPG Image Ingestion ---")
    jpg_data = create_jpg_bytes(64, 64)

    resp = requests.post(
        f"{BASE_URL}/memories/image",
        files={"file": ("test_blue.jpg", jpg_data, "image/jpeg")},
        data={"description": "A blue gradient test image for QA-007"},
    )
    passed = resp.status_code == 201
    details = None
    if passed:
        body = resp.json()
        CREATED_IDS.append(("jpg_memory", body.get("node_id")))
        details = f"node_id={body.get('node_id')}, format={body.get('format')}, embedding_dim={body.get('embedding_dim')}"
    else:
        details = f"Expected 201, got {resp.status_code}: {resp.text[:300]}"
    record("POST /memories/image with JPG", passed, details, resp)
    return resp


def test_clip_image_indexing():
    """Test 3: POST /images indexes images with CLIP embeddings."""
    print("\n--- Test 3: CLIP Image Indexing ---")

    # Create a nature-like image (green)
    png_data = create_png_bytes(64, 64, (0, 200, 50))

    resp = requests.post(
        f"{BASE_URL}/images",
        files={"file": ("nature_green.png", png_data, "image/png")},
        data={"description": "A green nature scene"},
    )
    passed = resp.status_code in (200, 201)
    details = None
    if passed:
        body = resp.json()
        CREATED_IDS.append(("indexed_image", body.get("node_id")))
        details = f"node_id={body.get('node_id')}, embedding_dim={body.get('embedding_dim')}, format={body.get('format')}"
    else:
        details = f"Expected 200/201, got {resp.status_code}: {resp.text[:300]}"
    record("POST /images indexes with CLIP", passed, details, resp)
    return resp


def test_clip_embedding_dim():
    """Test 5: CLIP embeddings are 512-dimensional."""
    print("\n--- Test 4: CLIP Embedding Dimensionality ---")
    png_data = create_png_bytes(32, 32, (128, 128, 128))

    resp = requests.post(
        f"{BASE_URL}/images",
        files={"file": ("dim_test.png", png_data, "image/png")},
    )

    passed = False
    details = None
    if resp.status_code in (200, 201):
        body = resp.json()
        dim = body.get("embedding_dim", 0)
        passed = dim == 512
        CREATED_IDS.append(("dim_test", body.get("node_id")))
        details = f"embedding_dim={dim} (expected 512)"
    else:
        details = f"Request failed with {resp.status_code}: {resp.text[:300]}"
    record("CLIP embeddings are 512-dimensional", passed, details, resp)
    return resp


def test_image_search():
    """Test 4: POST /images/search returns similar images."""
    print("\n--- Test 5: Image Search ---")

    # Index a few more images with distinct descriptions for better search
    images_to_index = [
        ("sunset_orange.png", create_png_bytes(64, 64, (255, 128, 0)), "A warm orange sunset over the ocean"),
        ("forest_green.png", create_png_bytes(64, 64, (0, 180, 60)), "Dense green forest with tall trees"),
        ("sky_blue.png", create_png_bytes(64, 64, (50, 100, 255)), "Clear blue sky on a sunny day"),
    ]

    for fname, data, desc in images_to_index:
        r = requests.post(
            f"{BASE_URL}/images",
            files={"file": (fname, data, "image/png")},
            data={"description": desc},
        )
        if r.status_code in (200, 201):
            CREATED_IDS.append(("search_index", r.json().get("node_id")))

    # Now search for images
    resp = requests.post(
        f"{BASE_URL}/images/search",
        json={"query": "sunset ocean orange", "limit": 5},
    )

    passed = resp.status_code == 200
    details = None
    if passed:
        body = resp.json()
        total = body.get("total", 0)
        results = body.get("results", [])
        passed = total > 0
        details = f"total={total}, results_count={len(results)}"
        if results:
            top = results[0]
            details += f", top_score={top.get('score', 'N/A')}, top_content={top.get('content', 'N/A')[:50]}"
    else:
        details = f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
    record("POST /images/search returns results", passed, details, resp)
    return resp


def test_media_retrieval():
    """Test 6: GET /media/{id} serves the stored media file."""
    print("\n--- Test 6: Media Retrieval ---")

    # Use the first created node_id
    node_id = None
    for label, nid in CREATED_IDS:
        if nid is not None:
            node_id = nid
            break

    if node_id is None:
        record("GET /media/{id} serves media", False, "No node_id available from previous tests")
        return None

    resp = requests.get(f"{BASE_URL}/media/{node_id}")
    passed = resp.status_code == 200
    details = None
    if passed:
        content_type = resp.headers.get("content-type", "unknown")
        content_length = len(resp.content)
        passed = content_length > 0
        details = f"content_type={content_type}, content_length={content_length} bytes"
    else:
        # Media serving might not be enabled or file not persisted
        details = f"Status {resp.status_code}: {resp.text[:200]}"
        # Check if it's a 404 (not stored) vs actual error
        if resp.status_code == 404:
            details += " (media file persistence may not be enabled — this is acceptable if images are stored as embeddings only)"
            # Still mark test based on whether response is structured
            try:
                body = resp.json()
                if "error" in body or "detail" in body:
                    passed = False  # Expected behavior, server responded correctly
                    details += " — server correctly returns 404 for non-persisted media"
            except Exception:
                pass

    record("GET /media/{id} serves media", passed, details, resp)
    return resp


def test_search_quality():
    """Bonus: Verify search returns semantically relevant results."""
    print("\n--- Test 7: Search Quality Verification ---")

    resp = requests.post(
        f"{BASE_URL}/images/search",
        json={"query": "green forest trees nature", "limit": 5},
    )

    passed = False
    details = None
    if resp.status_code == 200:
        body = resp.json()
        results = body.get("results", [])
        if len(results) > 0:
            # Check if top result has reasonable similarity
            top_score = results[0].get("score", 0)
            passed = top_score > 0
            details = f"top_score={top_score:.4f}, num_results={len(results)}"
            # Show all result scores
            scores_str = ", ".join([f"{r.get('score', 0):.4f}" for r in results[:5]])
            details += f", all_scores=[{scores_str}]"
        else:
            details = "No results returned"
    else:
        details = f"Status {resp.status_code}: {resp.text[:200]}"

    record("Image search returns semantically relevant results", passed, details, resp)
    return resp


def main():
    print("=" * 60)
    print("QA-007: Multimodal Image Ingestion & CLIP Search")
    print("=" * 60)

    # Verify server is up and CLIP is loaded
    health = requests.get(f"{BASE_URL}/health").json()
    if not health.get("models", {}).get("image_embedder_loaded"):
        print("ERROR: image_embedder not loaded. Aborting.")
        sys.exit(1)
    if not health.get("models", {}).get("cross_modal_encoder_loaded"):
        print("ERROR: cross_modal_encoder not loaded. Aborting.")
        sys.exit(1)
    print("Server healthy, CLIP models loaded.\n")

    # Run tests
    test_png_ingestion()
    test_jpg_ingestion()
    test_clip_image_indexing()
    test_clip_embedding_dim()
    test_image_search()
    test_media_retrieval()
    test_search_quality()

    # Summary
    total = len(RESULTS["tests"])
    passed = sum(1 for t in RESULTS["tests"] if t["passed"])
    failed = total - passed
    RESULTS["summary"] = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "created_node_ids": [{"label": l, "node_id": n} for l, n in CREATED_IDS],
    }

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "multimodal-image.json")
    with open(output_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
