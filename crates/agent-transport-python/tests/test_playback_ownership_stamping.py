"""播放邊界所有權復制(xbot DLG-029 契約的傳輸側半邊)。

對位原理:到達流周期邊界重建 —— sink 順序 == 到達順序,每個
TTSStoppedFrame 邊界之後的第一個音頻幀對應一個播放周期的開端。
覆核裁決的三個反例(流式重登記 off-by-one、vendor 每句一停、
3s 斷流 fallback)在此逐一入卷。
"""

from unittest.mock import AsyncMock

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from pipecat.frames.frames import SystemFrame

from agent_transport.sip.pipecat.sip_transport import (
    PlaybackOwnershipStamper,
    SipOutputTransport,
)


def _supersede(new_seq: int):
    """複刻 xbot.vendor.tts_ownership.PlaybackSupersedeFrame 的 metadata
    契約 —— 互通面就是 metadata,傳輸側不依賴類同一性。"""
    frame = SystemFrame()
    frame.metadata.update({
        "playback_control_version": 1,
        "new_turn_id": f"turn-{new_seq}",
        "new_turn_seq": new_seq,
        "cutoff_turn_seq": new_seq - 1,
    })
    return frame


def _audio(context_id: str, turn_id: str = "", *, versioned: bool = True):
    frame = TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=8000, num_channels=1)
    if versioned:
        frame.metadata["audio_ownership_version"] = 1
        frame.metadata["context_id"] = context_id
        if turn_id:
            frame.metadata["turn_id"] = turn_id
    return frame


def _cycle(stamper):
    """一個播放周期:成對 Started + 成對 Stopped,回傳四幀。"""
    frames = [
        BotStartedSpeakingFrame(), BotStartedSpeakingFrame(),
        BotStoppedSpeakingFrame(), BotStoppedSpeakingFrame(),
    ]
    for frame in frames:
        stamper.stamp(frame)
    return frames


def test_started_and_stopped_pairs_carry_context_ownership():
    stamper = PlaybackOwnershipStamper()
    stamper.observe_audio_frame(_audio("ctx-1", "turn-1"))

    frames = _cycle(stamper)

    for frame in frames:
        assert frame.metadata["audio_ownership_version"] == 1
        assert frame.metadata["context_id"] == "ctx-1"
        assert frame.metadata["turn_id"] == "turn-1"


def test_streaming_reregistration_does_not_shift_ownership():
    """回歸(覆核 P0 最小復現):context A 的快照被周期彈走後,A 仍在
    流式送幀 —— 不得被二次登記;B 的周期必須拿到 B。"""
    stamper = PlaybackOwnershipStamper()
    stamper.observe_audio_frame(_audio("ctx-1", "turn-1"))
    started = BotStartedSpeakingFrame()
    stamper.stamp(started)                      # A 周期開啟,彈走 A
    assert started.metadata["turn_id"] == "turn-1"
    stamper.observe_audio_frame(_audio("ctx-1", "turn-1"))   # A 流式後續幀
    stamper.observe_stop_frame()                # A 的 stop 抵達(周期邊界)
    stamper.observe_audio_frame(_audio("ctx-2", "turn-2"))
    stamper.stamp(BotStoppedSpeakingFrame())    # A 周期關閉

    second = BotStartedSpeakingFrame()
    stamper.stamp(second)
    assert second.metadata["turn_id"] == "turn-2"


def test_per_sentence_stops_same_context_keep_alignment():
    """回歸(覆核 P1,doubao/qwentts/minimax 形態):同 context 每句一個
    Stopped → 每句一個播放周期,每個周期都拿到本 context 的身份,
    排隊中的下一輪不被提前消費。"""
    stamper = PlaybackOwnershipStamper()
    # turn-1 兩句 + turn-2 一句,全部先到齊(深排隊)
    stamper.observe_audio_frame(_audio("ctx-1", "turn-1"))   # 句1
    stamper.observe_stop_frame()
    stamper.observe_audio_frame(_audio("ctx-1", "turn-1"))   # 句2:同 context 新周期
    stamper.observe_stop_frame()
    stamper.observe_audio_frame(_audio("ctx-2", "turn-2"))
    stamper.observe_stop_frame()

    first = _cycle(stamper)
    second = _cycle(stamper)
    third = _cycle(stamper)

    assert {f.metadata["turn_id"] for f in first} == {"turn-1"}
    assert {f.metadata["turn_id"] for f in second} == {"turn-1"}
    assert {f.metadata["turn_id"] for f in third} == {"turn-2"}


def test_fallback_resume_reuses_cycle_identity():
    """回歸(覆核 P1,pipecat 3s 斷流 fallback):MediaSender 內部補停後
    同段音頻續播 —— pending 為空的第二個 Started 沿用上一周期身份,
    不彈錯後續快照。"""
    stamper = PlaybackOwnershipStamper()
    stamper.observe_audio_frame(_audio("ctx-1", "turn-1"))
    stamper.stamp(BotStartedSpeakingFrame())    # 周期1:彈 A
    stamper.stamp(BotStoppedSpeakingFrame())    # fallback 補停(無到達流邊界)

    resumed = BotStartedSpeakingFrame()
    stamper.stamp(resumed)                      # 同 context 續播
    assert resumed.metadata["turn_id"] == "turn-1"
    assert resumed.metadata["context_id"] == "ctx-1"


def test_zero_audio_context_registers_nothing():
    """零音頻 context:只有 Stopped 沒有音頻 → 無周期、無快照,
    不佔用後續播放的所有權。"""
    stamper = PlaybackOwnershipStamper()
    stamper.observe_stop_frame()                # zero_audio 的 stop
    stamper.observe_audio_frame(_audio("ctx-2", "turn-2"))

    started = BotStartedSpeakingFrame()
    stamper.stamp(started)
    assert started.metadata["turn_id"] == "turn-2"


def test_unversioned_audio_takes_placeholder_not_next_identity():
    """無章音頻自成一個無主周期(空佔位),不偷走後續 context 的身份。"""
    stamper = PlaybackOwnershipStamper()
    stamper.observe_audio_frame(_audio("bypass", versioned=False))
    stamper.observe_stop_frame()
    stamper.observe_audio_frame(_audio("ctx-1", "turn-1"))

    bare = BotStartedSpeakingFrame()
    stamper.stamp(bare)                         # 無主周期:不蓋章
    assert "turn_id" not in bare.metadata
    assert "context_id" not in bare.metadata
    stamper.stamp(BotStoppedSpeakingFrame())

    owned = BotStartedSpeakingFrame()
    stamper.stamp(owned)
    assert owned.metadata["turn_id"] == "turn-1"


def test_unowned_system_speech_carries_context_without_turn_id():
    """開場白/結束語:mixin 蓋 context 但無 turn_id —— 播放事件同樣
    只帶 context,下游據此排除而非猜測。"""
    stamper = PlaybackOwnershipStamper()
    stamper.observe_audio_frame(_audio("greeting-ctx"))

    started = BotStartedSpeakingFrame()
    stamper.stamp(started)

    assert started.metadata["context_id"] == "greeting-ctx"
    assert "turn_id" not in started.metadata


def test_interruption_clears_pending_and_current_cycle():
    stamper = PlaybackOwnershipStamper()
    stamper.observe_audio_frame(_audio("ctx-1", "turn-1"))
    stamper.observe_interruption()

    started = BotStartedSpeakingFrame()
    stamper.stamp(started)

    assert "turn_id" not in started.metadata
    assert "context_id" not in started.metadata


def test_overflow_poisons_until_interruption():
    """回歸(覆核 P2):溢出時丟隊列任何一端都會鏈式錯位 —— 整隊清空
    並停止蓋章(寧缺勿錯),打斷後重新同步。"""
    stamper = PlaybackOwnershipStamper()
    for index in range(PlaybackOwnershipStamper._MAX_PENDING + 1):
        stamper.observe_audio_frame(_audio(f"ctx-{index}", f"turn-{index}"))
        stamper.observe_stop_frame()

    poisoned = BotStartedSpeakingFrame()
    stamper.stamp(poisoned)
    assert "turn_id" not in poisoned.metadata   # 中毒:不蓋章

    stamper.observe_interruption()              # 重新同步
    stamper.observe_audio_frame(_audio("ctx-x", "turn-x"))
    recovered = BotStartedSpeakingFrame()
    stamper.stamp(recovered)
    assert recovered.metadata["turn_id"] == "turn-x"


class _FakeEndpoint:
    output_sample_rate = 8000

    def __init__(self):
        self.cleared = []

    def clear_buffer(self, cid):
        self.cleared.append(cid)


def _make_output(monkeypatch):
    base_process = AsyncMock()
    monkeypatch.setattr(
        "pipecat.transports.base_output.BaseOutputTransport.process_frame",
        base_process,
    )
    monkeypatch.setattr(
        "pipecat.transports.base_output.BaseOutputTransport.push_frame",
        AsyncMock(),
    )
    endpoint = _FakeEndpoint()
    out = SipOutputTransport(endpoint, "sid-1", transport=None)
    return out, endpoint, base_process


# ─── DLG-030 輪次接管(supersede)傳輸側 ──────────────────────────────────


@pytest.mark.asyncio
async def test_supersede_gate_drops_stale_keeps_fresh_and_unowned(monkeypatch):
    """接管後:陳舊有主音頻被丟棄(不入 MediaSender、不登記),新輪次
    與無章音頻照常通行。"""
    out, endpoint, base_process = _make_output(monkeypatch)
    await out.process_frame(
        _supersede(4), FrameDirection.DOWNSTREAM
    )

    stale = _audio("ctx-3", "turn-3")
    await out.process_frame(stale, FrameDirection.DOWNSTREAM)
    fresh = _audio("ctx-4", "turn-4")
    await out.process_frame(fresh, FrameDirection.DOWNSTREAM)
    bare = _audio("bypass", versioned=False)
    await out.process_frame(bare, FrameDirection.DOWNSTREAM)

    forwarded = [call.args[0] for call in base_process.call_args_list]
    assert stale not in forwarded
    assert fresh in forwarded and bare in forwarded

    started = BotStartedSpeakingFrame()
    await out.push_frame(started, FrameDirection.UPSTREAM)
    assert started.metadata["turn_id"] == "turn-4"   # 陳舊幀未登記,對位不偏移


@pytest.mark.asyncio
async def test_supersede_clears_sink_only_with_stale_owned_content(monkeypatch):
    """sink 持有陳舊輪次音頻才清緩衝;僅無主播報(開場白)在放時不清。"""
    out, endpoint, _ = _make_output(monkeypatch)
    await out.process_frame(_audio("greeting-ctx"), FrameDirection.DOWNSTREAM)
    await out.process_frame(
        _supersede(1), FrameDirection.DOWNSTREAM
    )
    assert endpoint.cleared == []                    # 無主內容不受接管影響

    await out.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
    await out.process_frame(_audio("ctx-3", "turn-3"), FrameDirection.DOWNSTREAM)
    await out.process_frame(
        _supersede(4), FrameDirection.DOWNSTREAM
    )
    assert endpoint.cleared == ["sid-1"]             # 陳舊有主內容:清尾巴


@pytest.mark.asyncio
async def test_supersede_monotonic_late_signal_noop(monkeypatch):
    out, endpoint, _ = _make_output(monkeypatch)
    await out.process_frame(_audio("ctx-3", "turn-3"), FrameDirection.DOWNSTREAM)
    await out.process_frame(
        _supersede(4), FrameDirection.DOWNSTREAM
    )
    await out.process_frame(
        _supersede(3), FrameDirection.DOWNSTREAM
    )
    assert endpoint.cleared == ["sid-1"]             # 遲到信號不二次清

    stale = _audio("ctx-3", "turn-3")
    await out.process_frame(stale, FrameDirection.DOWNSTREAM)
    started = BotStartedSpeakingFrame()
    await out.push_frame(started, FrameDirection.UPSTREAM)
    assert "turn_id" not in started.metadata         # 權威序保持 4,turn-3 仍被擋


@pytest.mark.asyncio
async def test_supersede_residual_playback_degrades_to_unowned(monkeypatch):
    """清後仍從 MediaSender 滲出的陳舊殘餘:播放事件無主、被計量排除,
    絕不錯配到新輪次(缺身份勝於錯身份)。"""
    out, endpoint, _ = _make_output(monkeypatch)
    await out.process_frame(_audio("ctx-3", "turn-3"), FrameDirection.DOWNSTREAM)
    await out.process_frame(
        _supersede(4), FrameDirection.DOWNSTREAM
    )

    residual_started = BotStartedSpeakingFrame()     # 殘餘音頻觸發的播放事件
    await out.push_frame(residual_started, FrameDirection.UPSTREAM)
    assert "turn_id" not in residual_started.metadata
    await out.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

    await out.process_frame(_audio("ctx-4", "turn-4"), FrameDirection.DOWNSTREAM)
    owned = BotStartedSpeakingFrame()
    await out.push_frame(owned, FrameDirection.UPSTREAM)
    assert owned.metadata["turn_id"] == "turn-4"     # 新輪次身份不被殘餘污染


@pytest.mark.asyncio
async def test_output_transport_wiring(monkeypatch):
    """接線:process_frame 登記音頻/停止邊界/清隊,push_frame 蓋章
    播放事件,再交還 pipecat 基類。"""
    monkeypatch.setattr(
        "pipecat.transports.base_output.BaseOutputTransport.process_frame",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "pipecat.transports.base_output.BaseOutputTransport.push_frame",
        AsyncMock(),
    )
    endpoint = _FakeEndpoint()
    out = SipOutputTransport(endpoint, "sid-1", transport=None)

    await out.process_frame(_audio("ctx-1", "turn-1"), FrameDirection.DOWNSTREAM)
    started = BotStartedSpeakingFrame()
    await out.push_frame(started, FrameDirection.UPSTREAM)
    assert started.metadata["turn_id"] == "turn-1"
    assert started.metadata["context_id"] == "ctx-1"

    # 停止邊界經 process_frame 抵達 → 下一段音頻是新周期
    await out.process_frame(
        TTSStoppedFrame(), FrameDirection.DOWNSTREAM
    )
    await out.process_frame(_audio("ctx-2", "turn-2"), FrameDirection.DOWNSTREAM)
    await out.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    second = BotStartedSpeakingFrame()
    await out.push_frame(second, FrameDirection.UPSTREAM)
    assert second.metadata["turn_id"] == "turn-2"

    await out.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
    assert endpoint.cleared == ["sid-1"]
    stopped = BotStoppedSpeakingFrame()
    await out.push_frame(stopped, FrameDirection.UPSTREAM)
    assert "turn_id" not in stopped.metadata
