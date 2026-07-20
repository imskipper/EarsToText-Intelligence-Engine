import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "OmniAudio-Intelligence-Engine"

    # ASR runtime
    WHISPER_MODEL_SIZE: str = "small"
    ASR_DEVICE: str = "cpu"
    COMPUTE_TYPE: str = "float32"
    BEAM_SIZE: int = 5
    VAD_MIN_SILENCE_MS: int = 400
    ENABLE_WORD_TIMESTAMPS: bool = True

    # LLM orchestration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    INFERENCE_MODEL: str = "gpt-4o-mini"

    # Optional research backends. Keep disabled unless model access is configured.
    ENABLE_DIARIZATION: bool = False
    PYANNOTE_AUTH_TOKEN: str = os.getenv("PYANNOTE_AUTH_TOKEN", "")
    ENABLE_SUPERVISED_EMOTION: bool = False
    EMOTION_MODEL_ID: str = "superb/wav2vec2-base-superb-er"
    ENABLE_NATIVE_AUDIO_MODEL: bool = False
    NATIVE_AUDIO_MODEL_ID: str = ""
    ENABLE_STREAMING_PLACEHOLDER: bool = False

    @property
    def has_openai_api_key(self) -> bool:
        return bool(self.OPENAI_API_KEY and self.OPENAI_API_KEY != "mock-key")

    @property
    def has_diarization_backend(self) -> bool:
        return bool(self.ENABLE_DIARIZATION and self.PYANNOTE_AUTH_TOKEN)


settings = PipelineSettings()
