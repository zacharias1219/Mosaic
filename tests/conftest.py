import pytest

from mosaic.data.gen import generate
from mosaic.data.questions import build_questions
from mosaic.providers.sim import SimWorld


@pytest.fixture(scope="session")
def tickets_8k():
    return generate(8000, 42)


@pytest.fixture(scope="session")
def tickets_32k():
    return generate(32000, 42)


@pytest.fixture(scope="session")
def world_8k(tickets_8k):
    w = SimWorld()
    w.add_corpus(tickets_8k)
    for cls in "LAP":
        w.add_questions(build_questions(tickets_8k, cls, 20, 42, "8k"))
    return w
