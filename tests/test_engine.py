from transcription_sync import VisemeStream
from transcription_sync.visemes import CHANNELS


def test_render_produces_one_frame_per_letter():
    vs = VisemeStream()
    frames = vs.render_text("hello world", amplitude=0.3)
    # 10 letters (spaces/punct become pauses too, but "hello world" has 1 space)
    assert len(frames) == len("hello world")
    for f in frames:
        assert set(f.keys()) == set(CHANNELS)
        assert all(0.0 <= v <= 1.0 for f2 in frames for v in f2.values())


def test_silence_keeps_mouth_closed():
    vs = VisemeStream()
    frames = vs.render_text("aaa", amplitude=0.0)  # no loudness
    # With zero amplitude, scaled targets collapse; jaw should stay near shut.
    assert max(f["JawOpen"] for f in frames) < 0.05


def test_louder_opens_jaw_more():
    quiet = VisemeStream().render_text("aaaaa", amplitude=0.1)
    loud = VisemeStream().render_text("aaaaa", amplitude=0.4)
    assert max(f["JawOpen"] for f in loud) > max(f["JawOpen"] for f in quiet)


def test_bilabial_closes_lips():
    vs = VisemeStream()
    frames = vs.render_text("mmmm", amplitude=0.4)
    # /m/ has high MouthClose and ~zero JawOpen
    assert frames[-1]["MouthClose"] > frames[-1]["JawOpen"]


def test_streaming_matches_pending_count():
    vs = VisemeStream()
    vs.feed_text("hi")
    assert vs.pending == 2
    vs.set_amplitude(0.3)
    assert vs.tick() is not None
    assert vs.pending == 1
    vs.tick()
    assert vs.pending == 0
    assert vs.tick() is None


def test_reset_clears_state():
    vs = VisemeStream()
    vs.feed_text("abc")
    vs.reset()
    assert vs.pending == 0
    assert all(v == 0.0 for v in (vs.tick() or {"x": 0.0}).values()) or vs.tick() is None
