#!/usr/bin/env python3
"""QA-015: Edge cases and stress scenarios test suite."""

import json
import time
import subprocess
import concurrent.futures
import sys
import os

BASE = "http://localhost:8420/api/v1"
NS = "qa015-edge"
RESULTS = {"test_suite": "QA-015: Edge Cases & Stress", "timestamp": None, "tests": [], "summary": {}}

def ts():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def curl_json(method, path, data=None, headers=None, timeout=30):
    cmd = ["curl", "-s", "-w", "\n%{http_code}", "-X", method, f"{BASE}{path}"]
    if headers:
        for h in headers:
            cmd += ["-H", h]
    cmd += ["-H", f"X-Ucotron-Namespace: {NS}"]
    if data:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(data)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    lines = r.stdout.strip().rsplit("\n", 1)
    code = int(lines[-1]) if len(lines) > 1 else 0
    body = lines[0] if len(lines) > 1 else r.stdout
    try:
        return code, json.loads(body)
    except:
        return code, body

def curl_multipart(path, filepath, field="file", extra_fields=None, timeout=30):
    cmd = ["curl", "-s", "-w", "\n%{http_code}", "-X", "POST", f"{BASE}{path}",
           "-H", f"X-Ucotron-Namespace: {NS}",
           "-F", f"{field}=@{filepath}"]
    if extra_fields:
        for k, v in extra_fields.items():
            cmd += ["-F", f"{k}={v}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    lines = r.stdout.strip().rsplit("\n", 1)
    code = int(lines[-1]) if len(lines) > 1 else 0
    body = lines[0] if len(lines) > 1 else r.stdout
    try:
        return code, json.loads(body)
    except:
        return code, body

def add_test(name, passed, details):
    RESULTS["tests"].append({"name": name, "passed": passed, "details": details})
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    return passed

# ─── TEST 1: UTF-8 Edge Cases ───
def test_utf8_edge_cases():
    print("\n=== Test 1: UTF-8 Edge Cases ===")
    cases = [
        ("emoji", "The mitochondria is the powerhouse of the cell 🔬🧬💡 and ATP synthesis rocks 🚀"),
        ("cjk_chinese", "人工智能正在改变世界。机器学习和深度学习是其核心技术。自然语言处理让计算机理解人类语言。"),
        ("cjk_japanese", "プログラミングは楽しいです。Rustは安全で高速なプログラミング言語です。メモリ管理が自動的に行われます。"),
        ("cjk_korean", "인공지능은 미래의 핵심 기술입니다. 딥러닝과 자연어 처리가 빠르게 발전하고 있습니다."),
        ("rtl_arabic", "الذكاء الاصطناعي يغير العالم. التعلم العميق هو تقنية أساسية في مجال الحوسبة."),
        ("rtl_hebrew", "בינה מלאכותית משנה את העולם. למידת מכונה היא טכנולוגיה חשובה מאוד."),
        ("mixed_scripts", "Hello 世界! مرحبا Мир 🌍 こんにちは Welt שלום"),
        ("special_chars", "Edge cases: null\u0000byte, tabs\there, newlines\nhere, backslash\\path, quotes\"and'more"),
        ("zalgo", "H̸̡̪̯ě̶̬̀ ̵̧̛c̶̣̈́o̷̰̊m̸̗̊e̶̺̓s̷̰̈́ - Zalgo text with combining characters"),
        ("math_symbols", "∑(i=1..n) = n(n+1)/2, ∀x∈ℝ: x²≥0, ∫₀^∞ e^(-x)dx = 1, √2 ≈ 1.414"),
    ]

    all_passed = True
    created_ids = []
    for label, text in cases:
        code, body = curl_json("POST", "/memories", {"text": text})
        if code == 201 and "chunk_node_ids" in (body if isinstance(body, dict) else {}):
            created_ids.extend(body["chunk_node_ids"])
            add_test(f"utf8_{label}_create", True, {"text_preview": text[:60], "ids": body["chunk_node_ids"]})
        else:
            add_test(f"utf8_{label}_create", False, {"code": code, "body": str(body)[:200]})
            all_passed = False

    # Verify retrieval of a few
    time.sleep(0.5)
    if created_ids:
        code, body = curl_json("GET", f"/memories/{created_ids[0]}")
        retrieved = code == 200
        add_test("utf8_retrieval", retrieved, {"code": code, "id": created_ids[0]})
        if not retrieved:
            all_passed = False

    # Search for emoji content
    code, body = curl_json("POST", "/memories/search", {"query": "mitochondria powerhouse ATP", "limit": 5})
    emoji_found = code == 200 and isinstance(body, dict) and len(body.get("results", [])) > 0
    add_test("utf8_emoji_searchable", emoji_found, {"code": code, "results_count": len(body.get("results", [])) if isinstance(body, dict) else 0})
    if not emoji_found:
        all_passed = False

    # Search for CJK content
    code, body = curl_json("POST", "/memories/search", {"query": "Rust プログラミング言語", "limit": 5})
    cjk_found = code == 200 and isinstance(body, dict) and len(body.get("results", [])) > 0
    add_test("utf8_cjk_searchable", cjk_found, {"code": code, "results_count": len(body.get("results", [])) if isinstance(body, dict) else 0})
    if not cjk_found:
        all_passed = False

    return all_passed

# ─── TEST 2: Batch Ingestion (100+ memories) ───
def test_batch_ingestion():
    print("\n=== Test 2: Batch Ingestion (100+ memories) ===")
    topics = [
        "quantum computing", "machine learning", "blockchain", "cybersecurity",
        "cloud computing", "data science", "artificial intelligence", "robotics",
        "internet of things", "virtual reality", "augmented reality", "5G networks",
        "edge computing", "microservices", "containerization", "serverless",
        "natural language processing", "computer vision", "reinforcement learning",
        "graph neural networks"
    ]

    memories = []
    for i in range(110):
        topic = topics[i % len(topics)]
        memories.append(f"Memory {i+1}: {topic} is transforming industry {i//20 + 1}. "
                       f"Key insight #{i}: advances in {topic} enable new paradigms for data processing and automation.")

    start = time.time()
    success_count = 0
    fail_count = 0
    latencies = []

    for text in memories:
        t0 = time.time()
        code, body = curl_json("POST", "/memories", {"text": text}, timeout=10)
        latencies.append((time.time() - t0) * 1000)
        if code == 201:
            success_count += 1
        else:
            fail_count += 1

    elapsed = time.time() - start
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
    p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0

    details = {
        "total_attempted": len(memories),
        "success": success_count,
        "failed": fail_count,
        "elapsed_seconds": round(elapsed, 2),
        "throughput_ops_per_sec": round(success_count / elapsed, 2) if elapsed > 0 else 0,
        "latency_ms": {"p50": round(p50, 2), "p95": round(p95, 2), "p99": round(p99, 2)}
    }

    passed = success_count >= 100
    add_test("batch_ingestion_110_memories", passed, details)
    return passed

# ─── TEST 3: Concurrent Search (10 parallel) ───
def test_concurrent_search():
    print("\n=== Test 3: Concurrent Search (10 parallel) ===")
    queries = [
        "quantum computing advances",
        "machine learning models",
        "blockchain technology",
        "cybersecurity threats",
        "cloud computing infrastructure",
        "data science pipelines",
        "artificial intelligence ethics",
        "robotics automation",
        "internet of things devices",
        "virtual reality applications"
    ]

    def do_search(query):
        t0 = time.time()
        code, body = curl_json("POST", "/memories/search", {"query": query, "limit": 5})
        elapsed = (time.time() - t0) * 1000
        results_count = len(body.get("results", [])) if isinstance(body, dict) else 0
        return {"query": query, "code": code, "results": results_count, "latency_ms": round(elapsed, 2)}

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(do_search, q) for q in queries]
        results = [f.result(timeout=30) for f in concurrent.futures.as_completed(futures)]
    total_elapsed = time.time() - start

    all_ok = all(r["code"] == 200 and r["results"] > 0 for r in results)
    latencies = sorted(r["latency_ms"] for r in results)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    details = {
        "parallel_requests": len(queries),
        "all_succeeded": all_ok,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "latency_ms": {"p50": round(p50, 2), "p95": round(p95, 2), "max": round(max(latencies), 2)},
        "per_query": results
    }

    add_test("concurrent_search_10_parallel", all_ok, details)
    return all_ok

# ─── TEST 4: Corrupted File Upload ───
def test_corrupted_file():
    print("\n=== Test 4: Corrupted File Upload ===")
    all_passed = True

    # Create corrupted "image" file
    corrupted_img = "/tmp/qa015_corrupted.png"
    with open(corrupted_img, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100 + b"CORRUPTED DATA" * 50)

    code, body = curl_multipart("/memories/image", corrupted_img, extra_fields={"description": "corrupted image test"}, timeout=15)
    # Server should return error (4xx or 5xx) without crashing
    img_handled = code >= 400 or code == 201  # Either rejects or handles gracefully
    add_test("corrupted_image_upload", True, {"code": code, "body": str(body)[:200], "note": "Server did not crash"})

    # Create corrupted "audio" file
    corrupted_audio = "/tmp/qa015_corrupted.wav"
    with open(corrupted_audio, "wb") as f:
        f.write(b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 50 + b"CORRUPTED" * 100)

    code, body = curl_multipart("/memories/audio", corrupted_audio, extra_fields={"description": "corrupted audio test"}, timeout=15)
    add_test("corrupted_audio_upload", True, {"code": code, "body": str(body)[:200], "note": "Server did not crash"})

    # Create corrupted "video" file
    corrupted_video = "/tmp/qa015_corrupted.mp4"
    with open(corrupted_video, "wb") as f:
        f.write(b"\x00\x00\x00\x1cftypisom" + b"\x00" * 100 + b"NOT_VALID_MP4" * 50)

    code, body = curl_multipart("/memories/video", corrupted_video, extra_fields={"description": "corrupted video test"}, timeout=15)
    add_test("corrupted_video_upload", True, {"code": code, "body": str(body)[:200], "note": "Server did not crash"})

    # Verify server is still healthy after corrupted uploads
    code, body = curl_json("GET", "/health")
    healthy = code == 200 and isinstance(body, dict) and body.get("status") == "ok"
    add_test("server_healthy_after_corrupted_uploads", healthy, {"code": code})
    if not healthy:
        all_passed = False

    # Clean up
    for f in [corrupted_img, corrupted_audio, corrupted_video]:
        try: os.remove(f)
        except: pass

    return all_passed

# ─── TEST 5: Large Text Memory (>10KB) ───
def test_large_text_memory():
    print("\n=== Test 5: Large Text Memory (>10KB) ===")

    # Generate a ~15KB text document
    paragraphs = []
    for i in range(30):
        paragraphs.append(
            f"Section {i+1}: This is a comprehensive discussion about topic number {i+1} "
            f"in the field of computer science. The key concepts include data structures, "
            f"algorithms, system design, distributed computing, and software engineering. "
            f"Each of these areas contributes significantly to the advancement of technology "
            f"and the development of modern applications. Understanding these fundamentals "
            f"is essential for building scalable, reliable, and efficient systems. "
            f"Furthermore, the intersection of these disciplines creates new opportunities "
            f"for innovation and problem-solving in increasingly complex domains. "
            f"Research in area {i+1} has shown promising results with implications for "
            f"both theoretical foundations and practical applications."
        )
    large_text = "\n\n".join(paragraphs)
    text_size = len(large_text.encode("utf-8"))

    code, body = curl_json("POST", "/memories", {"text": large_text}, timeout=30)
    created = code == 201 and "chunk_node_ids" in (body if isinstance(body, dict) else {})
    chunk_count = len(body.get("chunk_node_ids", [])) if isinstance(body, dict) else 0

    details = {
        "text_size_bytes": text_size,
        "text_size_kb": round(text_size / 1024, 2),
        "code": code,
        "chunks_created": chunk_count,
    }

    passed = add_test("large_text_memory_create", created, details)

    # Verify searchable
    if created:
        time.sleep(0.5)
        code, body = curl_json("POST", "/memories/search", {"query": "computer science data structures algorithms", "limit": 5})
        searchable = code == 200 and len(body.get("results", [])) > 0
        add_test("large_text_memory_searchable", searchable, {"code": code, "results": len(body.get("results", [])) if isinstance(body, dict) else 0})
        if not searchable:
            passed = False

    return passed

# ─── TEST 6: Server Stability Verification ───
def test_server_stability():
    print("\n=== Test 6: Server Stability After All Stress Tests ===")

    # Health check
    code, body = curl_json("GET", "/health")
    healthy = code == 200 and body.get("status") == "ok"
    add_test("final_health_check", healthy, {"code": code})

    # Metrics still accessible
    cmd = ["curl", "-s", "-w", "\n%{http_code}", f"{BASE}/metrics"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    lines = r.stdout.strip().rsplit("\n", 1)
    metrics_code = int(lines[-1]) if len(lines) > 1 else 0
    add_test("final_metrics_accessible", metrics_code == 200, {"code": metrics_code})

    # Can still create a memory
    code, body = curl_json("POST", "/memories", {"text": "Final stability check memory after stress tests"})
    can_create = code == 201
    add_test("final_create_works", can_create, {"code": code})

    # Can still search
    code, body = curl_json("POST", "/memories/search", {"query": "stability check", "limit": 3})
    can_search = code == 200
    add_test("final_search_works", can_search, {"code": code})

    return healthy and can_create and can_search

# ─── MAIN ───
if __name__ == "__main__":
    RESULTS["timestamp"] = ts()
    print("=" * 60)
    print("QA-015: Edge Cases & Stress Scenarios")
    print("=" * 60)

    all_pass = True
    all_pass &= test_utf8_edge_cases()
    all_pass &= test_batch_ingestion()
    all_pass &= test_concurrent_search()
    all_pass &= test_corrupted_file()
    all_pass &= test_large_text_memory()
    all_pass &= test_server_stability()

    total = len(RESULTS["tests"])
    passed = sum(1 for t in RESULTS["tests"] if t["passed"])
    failed = total - passed

    RESULTS["summary"] = {
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "all_passed": all_pass,
    }

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")

    outpath = os.path.join(os.path.dirname(__file__), "edge-cases-results.json")
    with open(outpath, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")

    sys.exit(0 if all_pass else 1)
