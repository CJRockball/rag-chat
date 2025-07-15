from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # API Configuration
    api_title: str = "RAG Chat Application"
    api_version: str = "1.0.0"
    debug: bool = Field(default=False, description="Debug mode")

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")

    # Security
    secret_key: str = Field(..., description="Secret key for security")
    cors_origins: List[str] = Field(default=[],
                                    description="CORS allowed origins")

    # Google API
    google_api_key: str = Field(..., description="Google API key")

    # Database
    chroma_db_path: str = Field(
        default="src/utils/vectorstore/db_chroma",
        description="Chroma database path"
    )
    collection_name: str = Field(default="v_db",
                                 description="Collection name")

    # Document Processing
    doc_path: str = Field(..., description="Document path")

    # Rate Limiting
    rate_limit_requests_per_second: float = Field(
        default=0.1, description="Rate limit requests per second"
    )
    rate_limit_burst_size: int = Field(default=10,
                                       description="Rate limit burst size")

    # Logging
    log_level: str = Field(default="INFO", description="Log level")


settings = Settings()
