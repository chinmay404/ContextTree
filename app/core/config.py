import os
from typing import Any, Dict, List, Optional

from pydantic import validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    version_str: str = "v1"
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Convo Canvas API"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "Convo Canvas Backend API"
    
    # Server settings
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    DEBUG_MODE: bool = True
    LOG_LEVEL: str = "INFO"
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Database settings
    DATABASE_URL: Optional[str] = None
    
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()