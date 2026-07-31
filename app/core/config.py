"""Runtime configuration via environment variables (twelve-factor)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SEARCH_", env_file=".env")

    app_name: str = "semantic-search-service"
    version: str = "0.1.0"
    log_level: str = "INFO"

    #: Dimensionality of the default hashing embedder.
    embedding_dim: int = 512
    #: Load the bundled sample corpus on startup.
    load_sample_corpus: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
