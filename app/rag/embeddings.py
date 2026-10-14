"""
Embeddings module.

Provides the LangChain embedding model instance using Google Generative AI.
"""

import logging
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import EMBEDDING_MODEL, GEMINI_API_KEY, VECTOR_DIMENSION

logger = logging.getLogger(__name__)


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """
    Create and return a Google Generative AI embedding model.
    
    Uses gemini-embedding-001 with configurable output dimensionality.
    The default dimension is 768 (via MRL) for storage efficiency while
    maintaining high quality.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Cannot create embedding model.")

    # Set the API key in the environment for langchain-google-genai
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

    logger.info(
        "Initializing embedding model: %s (dimension=%d)",
        EMBEDDING_MODEL,
        VECTOR_DIMENSION,
    )

    kwargs = {}
    if VECTOR_DIMENSION:
        kwargs["output_dimensionality"] = VECTOR_DIMENSION

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY,
        **kwargs,
    )
