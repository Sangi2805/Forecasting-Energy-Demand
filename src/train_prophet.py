"""Backward-compatible entry point. Prefer: python -m src.prophet.train_prophet"""

from src.prophet.train_prophet import main

if __name__ == "__main__":
    main()
