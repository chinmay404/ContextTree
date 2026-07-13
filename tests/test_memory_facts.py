import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agent.memory import extract_memory_facts


def _name(text: str):
    facts = extract_memory_facts(None, [{"role": "user", "content": text}])
    return facts["user_profile"].get("name")


def test_extracts_name_from_im_contraction():
    assert _name("Hi I'm Chinmay") == "Chinmay"


def test_extracts_name_from_i_am():
    assert _name("Actually I am Chinmay, nice to meet you") == "Chinmay"


def test_still_extracts_from_my_name_is():
    assert _name("my name is Chinmay") == "Chinmay"


def test_does_not_capture_lowercase_verb_after_im():
    # "I'm going" must not be read as the name "going".
    assert _name("I'm going to the store") is None
