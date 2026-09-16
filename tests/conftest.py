import pytest

from aftersales_rag.agent import Assistant
from aftersales_rag.config import Settings


@pytest.fixture(scope="session")
def assistant() -> Assistant:
    return Assistant(Settings(anthropic_api_key=None))
