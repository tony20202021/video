from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    BOT_TOKEN: str
    BACKEND_URL: str = "http://localhost:8000"
    ADMIN_IDS: str = ""

    def is_admin(self, user_id: int) -> bool:
        return user_id in self._admin_ids_list()

    def _admin_ids_list(self) -> list[int]:
        return [int(x.strip()) for x in self.ADMIN_IDS.split(",") if x.strip()]

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
