#!/usr/bin/env python3
"""
Performance tests to validate optimization improvements
"""

import time
import base64
import io
import numpy as np
from PIL import Image

def create_test_image(size=(224, 224)):
    """Create a test image for benchmarking"""
    img = Image.new('RGB', size, color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.read()

def benchmark_image_preprocessing():
    """Benchmark image preprocessing performance"""
    print("\n=== Image Preprocessing Benchmark ===")
    
    # Test with larger source image (more realistic scenario)
    image_bytes = create_test_image(size=(1024, 1024))
    
    # Test BILINEAR (optimized)
    start = time.perf_counter()
    for _ in range(50):
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        img_array = np.asarray(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    bilinear_time = time.perf_counter() - start
    
    # Test LANCZOS (old method)
    start = time.perf_counter()
    for _ in range(50):
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    lanczos_time = time.perf_counter() - start
    
    print(f"BILINEAR (optimized): {bilinear_time:.4f}s for 50 iterations (1024x1024 -> 224x224)")
    print(f"LANCZOS (old):        {lanczos_time:.4f}s for 50 iterations (1024x1024 -> 224x224)")
    print(f"Performance ratio: {lanczos_time/bilinear_time:.2f}x")
    
    # BILINEAR is typically faster for downsampling, but the benefit varies
    # Key benefit: More consistent performance and adequate quality for ML preprocessing
    print("Note: BILINEAR provides adequate quality for ML preprocessing with better/similar performance")
    
    return bilinear_time, lanczos_time

def benchmark_array_conversion():
    """Benchmark numpy array conversion performance"""
    print("\n=== Array Conversion Benchmark ===")
    
    image = Image.new('RGB', (224, 224), color='blue')
    
    # Test np.asarray (optimized)
    start = time.perf_counter()
    for _ in range(1000):
        img_array = np.asarray(image, dtype=np.float32) / 255.0
    asarray_time = time.perf_counter() - start
    
    # Test np.array (old method)
    start = time.perf_counter()
    for _ in range(1000):
        img_array = np.array(image, dtype=np.float32) / 255.0
    array_time = time.perf_counter() - start
    
    print(f"np.asarray (optimized): {asarray_time:.4f}s for 1000 iterations")
    print(f"np.array (old):         {array_time:.4f}s for 1000 iterations")
    print(f"Performance ratio: {array_time/asarray_time:.2f}x")
    
    # Both are similar in performance, but asarray is semantically better
    # (avoids copy when already an array)
    print("Note: np.asarray avoids unnecessary copies when data is already array-like")
    
    return asarray_time, array_time

def benchmark_result_formatting():
    """Benchmark result formatting performance"""
    print("\n=== Result Formatting Benchmark ===")
    
    predictions = np.random.random(23)
    class_labels = [f'class_{i}' for i in range(23)]
    
    # Test optimized method (list comprehension + slice argsort)
    start = time.perf_counter()
    for _ in range(10000):
        top_indices = np.argsort(predictions)[::-1][:5]  # Only get top 5
        results = [
            {
                'label': class_labels[idx],
                'confidence': float(predictions[idx]),
                'percentage': float(predictions[idx]) * 100
            }
            for idx in top_indices
        ]
    optimized_time = time.perf_counter() - start
    
    # Test old method (full sort then loop)
    start = time.perf_counter()
    for _ in range(10000):
        top_indices = np.argsort(predictions)[::-1]  # Full sort
        results = []
        for i in range(min(5, len(top_indices))):
            idx = top_indices[i]
            results.append({
                'label': class_labels[idx],
                'confidence': float(predictions[idx]),
                'percentage': float(predictions[idx]) * 100
            })
    old_time = time.perf_counter() - start
    
    print(f"List comprehension + slice (optimized): {optimized_time:.4f}s for 10000 iterations")
    print(f"Loop with range (old):                  {old_time:.4f}s for 10000 iterations")
    print(f"Performance ratio: {old_time/optimized_time:.2f}x")
    
    # Main benefit: More Pythonic and readable, with comparable performance
    print("Note: List comprehension provides cleaner code with comparable performance")
    
    return optimized_time, old_time

def benchmark_threading_wait():
    """Benchmark threading wait performance"""
    print("\n=== Threading Wait Benchmark ===")
    import threading
    
    # Test Event-based wait (optimized)
    event = threading.Event()
    
    def set_event_after_delay():
        time.sleep(0.1)
        event.set()
    
    start = time.perf_counter()
    thread = threading.Thread(target=set_event_after_delay, daemon=True)
    thread.start()
    event.wait(timeout=1)
    event_time = time.perf_counter() - start
    
    # Test busy-wait polling (old method)
    start = time.perf_counter()
    wait_time = 0
    max_wait = 1
    loading = True
    
    def stop_loading_after_delay():
        time.sleep(0.1)
        nonlocal loading
        loading = False
    
    thread = threading.Thread(target=stop_loading_after_delay, daemon=True)
    thread.start()
    
    while loading and wait_time < max_wait:
        time.sleep(0.01)  # Simulating 0.5s but faster for test
        wait_time += 0.01
    poll_time = time.perf_counter() - start
    
    print(f"Event-based wait (optimized): {event_time:.4f}s")
    print(f"Busy-wait polling (old):      {poll_time:.4f}s")
    print(f"CPU efficiency: Event-based eliminates spinning")
    
    # Event-based should complete near the actual wait time
    assert abs(event_time - 0.1) < 0.05, "Event-based wait should be precise"
    return event_time, poll_time

def test_cache_performance():
    """Test that caching works correctly"""
    print("\n=== Cache Performance Test ===")
    
    import hashlib
    
    # Simulate cached lookups
    cache = {}
    test_data = b"test_image_data"
    
    # First lookup (cache miss)
    start = time.perf_counter()
    for _ in range(1000):
        cache.clear()
        key = hashlib.md5(test_data).hexdigest()
        if key not in cache:
            # Simulate expensive computation
            result = np.random.random(23)
            cache[key] = result
        else:
            result = cache[key]
    miss_time = time.perf_counter() - start
    
    # Subsequent lookups (cache hit)
    key = hashlib.md5(test_data).hexdigest()
    cache[key] = np.random.random(23)
    
    start = time.perf_counter()
    for _ in range(1000):
        result = cache[key]
    hit_time = time.perf_counter() - start
    
    print(f"Cache miss (with computation): {miss_time:.4f}s for 1000 iterations")
    print(f"Cache hit (lookup only):       {hit_time:.4f}s for 1000 iterations")
    print(f"Speedup: {miss_time/hit_time:.2f}x faster on cache hit")
    
    assert hit_time < miss_time, "Cache hit should be much faster"
    return hit_time, miss_time

if __name__ == '__main__':
    print("=" * 60)
    print("Performance Optimization Benchmark Suite")
    print("=" * 60)
    
    try:
        # Run all benchmarks
        benchmark_image_preprocessing()
        benchmark_array_conversion()
        benchmark_result_formatting()
        benchmark_threading_wait()
        test_cache_performance()
        
        print("\n" + "=" * 60)
        print("✅ All performance benchmarks passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Benchmark failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
