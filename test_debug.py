#!/usr/bin/env python3
"""
Simple test file to verify Python debugging works
"""

def test_function():
    """Test function for debugging"""
    x = 10
    y = 20
    result = x + y
    print(f"x = {x}, y = {y}, result = {result}")
    return result

def main():
    """Main function"""
    print("Starting debug test...")
    
    # Set breakpoint here
    value = test_function()
    
    print(f"Test completed. Final value: {value}")
    return value

if __name__ == "__main__":
    main()
