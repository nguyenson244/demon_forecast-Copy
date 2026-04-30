"""
Entry point duy nhất để chạy toàn bộ hệ thống dự báo.
Bạn chỉ cần chạy file này từ thư mục gốc:
    python run_pipeline.py
"""

import os
import sys
from pathlib import Path

# Thêm thư mục hiện tại vào sys.path để Python nhận diện được package 'src'
root_path = str(Path(__file__).parent.absolute())
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.pipeline.pipeline import run_pipeline
from src.utils.logger import get_logger

logger = get_logger("Runner")

if __name__ == "__main__":
    try:
        # Mặc định chạy toàn bộ từ bước 1 đến 5
        run_pipeline(start_step=1, end_step=5)
    except KeyboardInterrupt:
        logger.warning("\nStopped by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
