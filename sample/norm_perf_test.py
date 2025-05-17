#!/usr/bin/env python
# Test script to verify the performance improvement in cv2.norm(img1, img2)

import cv2
import numpy as np
import time

def main():
    # Generate test matrices
    print("Generating test matrices...")
    width, height = 1024, 768
    img1 = np.random.rand(height, width, 3).astype(np.float32)
    img2 = np.random.rand(height, width, 3).astype(np.float32)
    
    print(f"Testing with matrices of shape {img1.shape}, type {img1.dtype}")
    
    # Parameters
    num_iterations = 50
    
    # Test both versions of the norm
    print("\nTesting cv2.norm(img1, img2)...")
    t_start = time.time()
    result1 = 0
    for _ in range(num_iterations):
        result1 = cv2.norm(img1, img2, cv2.NORM_L2)
    t_direct = (time.time() - t_start) / num_iterations
    
    print(f"Direct method time: {t_direct*1000:.2f} ms, result: {result1:.2f}")
    
    print("\nTesting cv2.norm(img1 - img2)...")
    t_start = time.time()
    result2 = 0
    for _ in range(num_iterations):
        result2 = cv2.norm(img1 - img2, cv2.NORM_L2)
    t_subtract = (time.time() - t_start) / num_iterations
    
    print(f"Subtraction method time: {t_subtract*1000:.2f} ms, result: {result2:.2f}")
    
    # Compare performance and results
    ratio = t_direct / t_subtract
    print(f"\nPerformance ratio (direct/subtract): {ratio:.2f}")
    print(f"Results match within 1e-5: {abs(result1 - result2) < 1e-5}")
    
    # For a healthy implementation, the direct method should be similar or faster
    # than the subtraction method (ratio >= 1.0)
    if ratio >= 0.9:
        print("\nTest PASSED: Direct method has good performance")
    else:
        print("\nTest FAILED: Direct method is significantly slower than subtraction method")
    
    # Also test L1 norm
    print("\n--- Testing L1 norm ---")
    t_start = time.time()
    for _ in range(num_iterations):
        result1 = cv2.norm(img1, img2, cv2.NORM_L1)
    t_direct = (time.time() - t_start) / num_iterations
    
    t_start = time.time()
    for _ in range(num_iterations):
        result2 = cv2.norm(img1 - img2, cv2.NORM_L1)
    t_subtract = (time.time() - t_start) / num_iterations
    
    ratio = t_direct / t_subtract
    print(f"L1 norm - Performance ratio (direct/subtract): {ratio:.2f}")
    print(f"L1 norm - Results match within 1e-5: {abs(result1 - result2) < 1e-5}")
    
    # Also test INF norm
    print("\n--- Testing INF norm ---")
    t_start = time.time()
    for _ in range(num_iterations):
        result1 = cv2.norm(img1, img2, cv2.NORM_INF)
    t_direct = (time.time() - t_start) / num_iterations
    
    t_start = time.time()
    for _ in range(num_iterations):
        result2 = cv2.norm(img1 - img2, cv2.NORM_INF)
    t_subtract = (time.time() - t_start) / num_iterations
    
    ratio = t_direct / t_subtract
    print(f"INF norm - Performance ratio (direct/subtract): {ratio:.2f}")
    print(f"INF norm - Results match within 1e-5: {abs(result1 - result2) < 1e-5}")

if __name__ == "__main__":
    main()