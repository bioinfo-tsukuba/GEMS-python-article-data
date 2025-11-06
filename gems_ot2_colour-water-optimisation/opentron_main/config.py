import os
from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    host: str = Field(...)
    port: int = Field(22)
    username: str = Field(...)
    key_path: str = Field(...)
    ot2_code_file: str = Field(...)
    destination_dir: str = Field("/data/user_storage")
    key_type: str = Field(...)
    password: str | None = Field(None)
    ot2_calibration_file: str | None = Field(None)

    debug: bool = Field(False)

    class Config:
        env_file = '.env'

# 設定のインスタンスを作成
config = Config()
# config.model_post_init(__context=None)
