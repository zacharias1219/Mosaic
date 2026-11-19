"""9.5 Response cache."""
from mosaic.providers.base import Completion
from mosaic.providers.cache import ResponseCache, cache_key

MSG = [{"role": "user", "content": "hi"}]


def comp(**kw):
    return Completion(**{"text": "{}", "tokens_in": 3, "tokens_out": 2, "ttft_ms": 120.0, "total_ms": 300.0,
                         "queue_ms": 50.0, **kw})


def test_identical_requests_hit_any_change_misses(tmp_path):
    c = ResponseCache(tmp_path / "c.sqlite")
    k = cache_key("groq", "m", MSG, {"max_tokens": 10, "temperature": 0})
    c.put(k, "groq", "m", comp())
    assert c.get(cache_key("groq", "m", MSG, {"temperature": 0, "max_tokens": 10})) is not None
    for other in (cache_key("groq", "m", MSG, {"max_tokens": 11, "temperature": 0}),
                  cache_key("cerebras", "m", MSG, {"max_tokens": 10, "temperature": 0}),
                  cache_key("groq", "m2", MSG, {"max_tokens": 10, "temperature": 0}),
                  cache_key("groq", "m", [{"role": "user", "content": "hi!"}], {"max_tokens": 10, "temperature": 0}),
                  cache_key("groq", "m", MSG, {"max_tokens": 10, "temperature": 0}, schema={"a": 1})):
        assert c.get(other) is None


def test_replay_returns_original_timings_flagged(tmp_path):
    c = ResponseCache(tmp_path / "c.sqlite")
    c.put("k", "groq", "m", comp())
    r = c.get("k")
    assert r.replayed and r.ttft_ms == 120.0 and r.total_ms == 300.0 and r.queue_ms == 50.0


def test_no_cache_bypasses_reads_but_writes(tmp_path):
    c = ResponseCache(tmp_path / "c.sqlite", read=False, write=True)
    c.put("k", "groq", "m", comp())
    assert c.get("k") is None
    assert ResponseCache(tmp_path / "c.sqlite").get("k") is not None


def test_survives_restart(tmp_path):
    c = ResponseCache(tmp_path / "c.sqlite")
    c.put("k", "groq", "m", comp(text='{"x": 1}'))
    c.close()
    c2 = ResponseCache(tmp_path / "c.sqlite")
    assert c2.get("k").text == '{"x": 1}' and len(c2) == 1
