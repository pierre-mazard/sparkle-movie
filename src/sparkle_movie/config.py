from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import os


load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("SPARK_APP_NAME", "sparkle-movie")
    spark_master: str = os.getenv("SPARK_MASTER", "local[*]")
    movielens_size: str = os.getenv("MOVIELENS_SIZE", "ml-latest-small")
    movielens_data_dir: Path = Path(os.getenv("MOVIELENS_DATA_DIR", "data/raw/ml-latest-small"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "data/processed"))
    figures_dir: Path = Path(os.getenv("FIGURES_DIR", "reports/figures"))


settings = Settings()
