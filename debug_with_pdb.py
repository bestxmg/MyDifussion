#!/usr/bin/env python3
"""
Simple debugging script using pdb directly
This bypasses VS Code Python extension issues
"""

import pdb
import sys

def main():
    print("🐛 Starting debug session...")
    
    # Set a breakpoint
    pdb.set_trace()
    
    x = 10
    y = 20
    result = x + y
    
    print(f"x = {x}, y = {y}, result = {result}")
    
    # Another breakpoint
    pdb.set_trace()
    
    print("Debug session completed!")

if __name__ == "__main__":
    main()
