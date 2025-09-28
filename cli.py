#!/usr/bin/env python3
"""
Hockey Analyzer CLI Entry Point
"""

import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from hockey_analyzer.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
