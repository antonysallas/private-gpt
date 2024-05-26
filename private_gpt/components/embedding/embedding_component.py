import logging

from injector import inject, singleton
from llama_index.core.embeddings import BaseEmbedding, MockEmbedding
from private_gpt.paths import models_cache_path
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

@singleton
class EmbeddingComponent:
    embedding_model: BaseEmbedding

    @inject
    def __init__(self, settings: Settings) -> None:
        embedding_mode = settings.embedding.mode
        logger.info("Initializing the embedding model in mode=%s", embedding_mode)

        match embedding_mode:
            case "huggingface":
                self._initialize_huggingface(settings)
            case "sagemaker":
                self._initialize_sagemaker(settings)
            case "openai":
                self._initialize_openai(settings)
            case "ollama":
                self._initialize_ollama(settings)
            case "azopenai":
                self._initialize_azopenai(settings)
            case "mock":
                self.embedding_model = MockEmbedding(384)
            case _:
                raise ValueError(f"Unknown embedding mode: {embedding_mode}")

    def _initialize_huggingface(self, settings: Settings) -> None:
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        except ImportError as e:
            logger.error("Error importing HuggingFaceEmbedding: %s", e)
            raise ImportError(
                "Local dependencies not found, install with `poetry install --extras embeddings-huggingface`"
            ) from e

        self.embedding_model = HuggingFaceEmbedding(
            model_name=settings.huggingface.embedding_hf_model_name,
            cache_folder=str(models_cache_path),
        )

    def _initialize_sagemaker(self, settings: Settings) -> None:
        try:
            from private_gpt.components.embedding.custom.sagemaker import SagemakerEmbedding
        except ImportError as e:
            logger.error("Error importing SagemakerEmbedding: %s", e)
            raise ImportError(
                "Sagemaker dependencies not found, install with `poetry install --extras embeddings-sagemaker`"
            ) from e

        self.embedding_model = SagemakerEmbedding(
            endpoint_name=settings.sagemaker.embedding_endpoint_name,
        )

    def _initialize_openai(self, settings: Settings) -> None:
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
        except ImportError as e:
            logger.error("Error importing OpenAIEmbedding: %s", e)
            raise ImportError(
                "OpenAI dependencies not found, install with `poetry install --extras embeddings-openai`"
            ) from e

        self.embedding_model = OpenAIEmbedding(api_key=settings.openai.api_key)

    def _initialize_ollama(self, settings: Settings) -> None:
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding
        except ImportError as e:
            logger.error("Error importing OllamaEmbedding: %s", e)
            raise ImportError(
                "Local dependencies not found, install with `poetry install --extras embeddings-ollama`"
            ) from e

        self.embedding_model = OllamaEmbedding(
            model_name=settings.ollama.embedding_model,
            base_url=settings.ollama.embedding_api_base,
        )

    def _initialize_azopenai(self, settings: Settings) -> None:
        try:
            from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
        except ImportError as e:
            logger.error("Error importing AzureOpenAIEmbedding: %s", e)
            raise ImportError(
                "Azure OpenAI dependencies not found, install with `poetry install --extras embeddings-azopenai`"
            ) from e

        self.embedding_model = AzureOpenAIEmbedding(
            model=settings.azopenai.embedding_model,
            deployment_name=settings.azopenai.embedding_deployment_name,
            api_key=settings.azopenai.api_key,
            azure_endpoint=settings.azopenai.azure_endpoint,
            api_version=settings.azopenai.api_version,
        )
