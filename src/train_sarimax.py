"""Backward-compatible entry point. Prefer: python -m src.sarimax.train_sarimax"""

from src.sarimax.train_sarimax import main

if __name__ == "__main__":
    main()
