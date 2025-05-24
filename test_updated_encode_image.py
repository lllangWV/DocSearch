#!/usr/bin/env python3
"""
Test script for the updated encode_image function in image_processing.py
"""

from pathlib import Path

import numpy as np
from PIL import Image

from docsearch.image_processing import encode_image


def test_encode_image():
    """Test the updated encode_image function with different input types."""

    print("=== Testing Updated encode_image Function ===")

    # Test 1: PIL Image input
    print("\n1. Testing PIL Image input:")
    pil_image = Image.new("RGB", (100, 100), color="red")
    pil_b64 = encode_image(pil_image)
    print(f"PIL Image -> base64: {len(pil_b64)} characters")
    print(f"Starts with: {pil_b64[:50]}...")

    # Test 2: Numpy array input (RGB)
    print("\n2. Testing numpy array input (RGB):")
    rgb_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    rgb_b64 = encode_image(rgb_array)
    print(f"RGB numpy array -> base64: {len(rgb_b64)} characters")
    print(f"Starts with: {rgb_b64[:50]}...")

    # Test 3: Numpy array input (grayscale)
    print("\n3. Testing numpy array input (grayscale):")
    gray_array = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    gray_b64 = encode_image(gray_array)
    print(f"Grayscale numpy array -> base64: {len(gray_b64)} characters")
    print(f"Starts with: {gray_b64[:50]}...")

    # Test 4: Float numpy array (0-1 range)
    print("\n4. Testing float numpy array (0-1 range):")
    float_array = np.random.rand(30, 30, 3)
    float_b64 = encode_image(float_array)
    print(f"Float numpy array -> base64: {len(float_b64)} characters")
    print(f"Starts with: {float_b64[:50]}...")

    # Test 5: Create test image file and test path input
    print("\n5. Testing file path input:")
    test_image_path = Path("test_img.png")

    # Create test image
    test_img = Image.new("RGB", (40, 40), color="blue")
    test_img.save(test_image_path)

    try:
        # Test string path
        path_b64_str = encode_image(str(test_image_path))
        print(f"Image file (str path) -> base64: {len(path_b64_str)} characters")

        # Test Path object
        path_b64_path = encode_image(test_image_path)
        print(f"Image file (Path object) -> base64: {len(path_b64_path)} characters")

        # Verify they're the same
        print(f"String and Path results match: {path_b64_str == path_b64_path}")

    finally:
        # Cleanup
        if test_image_path.exists():
            test_image_path.unlink()

    # Test 6: Different formats
    print("\n6. Testing different output formats:")
    test_image = Image.new("RGB", (30, 30), color="green")

    png_b64 = encode_image(test_image, format="PNG")
    jpeg_b64 = encode_image(test_image, format="JPEG")

    print(f"PNG format: {len(png_b64)} characters")
    print(f"JPEG format: {len(jpeg_b64)} characters")

    # Test 7: RGBA -> JPEG conversion
    print("\n7. Testing RGBA -> JPEG auto-conversion:")
    rgba_image = Image.new("RGBA", (30, 30), color=(255, 0, 0, 128))
    rgba_jpeg_b64 = encode_image(rgba_image, format="JPEG")
    print(f"RGBA -> JPEG: {len(rgba_jpeg_b64)} characters (auto-converted to RGB)")

    print("\n=== Error Handling Tests ===")

    # Test error cases
    try:
        # 4D array
        invalid_array = np.random.rand(10, 10, 10, 10)
        encode_image(invalid_array)
    except TypeError as e:
        print(f"✓ 4D array error: {e}")

    try:
        # Invalid type
        encode_image(12345)
    except ValueError as e:
        print(f"✓ Invalid type error: {e}")

    try:
        # Non-existent file
        encode_image("nonexistent.png")
    except FileNotFoundError as e:
        print(f"✓ File not found error: {e}")

    print("\n=== Test completed successfully! ===")


if __name__ == "__main__":
    test_encode_image()
