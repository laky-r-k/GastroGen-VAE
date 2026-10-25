#!/usr/bin/env python3
"""
Convenience entry point at project root.
Delegates to experiments.run:main().
Usage:
    python run.py --exp experiment1
    python run.py --list
"""

from experiments.run import main

if __name__ == "__main__":
    main()
