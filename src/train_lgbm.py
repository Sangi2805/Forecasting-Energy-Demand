"""Backward-compatible entry point. Prefer: python -m src.lgbm.train_lgbm"""

from src.lgbm.train_lgbm import main

if __name__ == "__main__":
    main()
