"""Pipecat BaseTransport adapter for SIP transport.

Replaces Pipecat's WebsocketServerTransport for SIP calls.
All audio codec/resampling/pacing is handled in Rust. Python only bridges frames.

100% compatible with Pipecat's transport interface. Exposes all SIP features:
- Event handlers: on_client_connected, on_client_disconnected, on_beep_detected, on_beep_timeout
- Session metadata: session_id, remote_uri, direction, extra_headers
- SIP call control: transfer, hold, reject, beep detection
- Audio: mute, recording, background audio, flush/playout

Frame handling:
- OutputAudioRawFrame → send_audio_notify (Rust backpressure + 20ms RTP pacing)
- InterruptionFrame → clear_buffer (local RTP send-buffer drop; no network
  message — cheap, tightens barge-in latency. See process_frame below)
- OutputDTMFFrame → send_dtmf (RFC 2833 or SIP INFO)
- OutputTransportMessageFrame → send_info (SIP INFO with JSON body)
- EndFrame/CancelFrame → hangup
- InputAudioRawFrame ← recv_audio_bytes_blocking (Rust decodes + resamples)
- InputDTMFFrame ← event polling

Bot speaking state (BotStartedSpeaking/BotStoppedSpeaking) is handled by
Pipecat's BaseOutputTransport MediaSender infrastructure.
"""

import asyncio
import json

from collections import deque
from typing import Any, Dict, Optional

from loguru import logger

from agent_transport._event_sink import _on_event_from_rust
from agent_transport._executors import audio_io_executor
from agent_transport._ffi_queue import GLOBAL_DICT

try:
    from pipecat.audio.dtmf.types import KeypadEntry
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame, BotStoppedSpeakingFrame,
        CancelFrame, EndFrame, Frame, InputAudioRawFrame,
        InputDTMFFrame, InterruptionFrame, OutputAudioRawFrame,
        StartFrame, TTSAudioRawFrame, TTSStoppedFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection
    from pipecat.transports.base_input import BaseInputTransport
    from pipecat.transports.base_output import BaseOutputTransport
    from pipecat.transports.base_transport import BaseTransport, TransportParams
except ImportError:
    raise ImportError("pipecat-ai is required: pip install pipecat-ai")


# ─── Playback ownership ─────────────────────────────────────────────────────


def _turn_seq(turn_id: object) -> Optional[int]:
    """canonical turn_id 形如 ``turn-<n>``;不可解析視為無主,永不作廢。"""
    text = str(turn_id or "")
    if text.startswith("turn-") and text[5:].isdigit():
        return int(text[5:])
    return None


def _owned_seq(metadata: dict) -> Optional[int]:
    """音頻幀的輪次序號:優先 xbot mixin 蓋的數字 ``turn_seq``,
    回退解析 ``turn_id``;兩者皆無 = 無主,永不作廢。"""
    seq = metadata.get("turn_seq")
    if isinstance(seq, int) and seq > 0:
        return seq
    return _turn_seq(metadata.get("turn_id"))


# DLG-030 播放接管契約(xbot 側發射,見 xbot.vendor.tts_ownership 的
# PlaybackSupersedeFrame):互通面是 metadata,不依賴跨倉類同一性 ——
# 信號幀 metadata 攜帶 ``playback_control_version`` / ``new_turn_seq`` /
# ``cutoff_turn_seq``,SystemFrame 帶外傳播(不排在要作廢的音頻後面)。
# 序號 <= cutoff 的有主音頻失去播放權。
#
# 職責邊界(實測釘死):MediaSender 切塊會重建幀、丟棄 metadata(pipecat
# base_output ``handle_audio_frame``),Rust sink 容量僅 ~400ms
# (audio_buffer capacity = 2×200ms 閾值)——已在 MediaSender 隊列裡的
# 陳舊大頭無身份可辨,其取消(TTS context 取消 + 打斷機制沖刷 + 分級
# 停止策略)屬 xbot 側職責。傳輸側做最後一英里:丟棄後續到達的失權
# 音頻(含 DLG-029 記錄的 clear/cancel 縫隙漏網幀)+ 條件清 Rust 尾巴。
_PLAYBACK_CONTROL_VERSION_KEY = "playback_control_version"
_PLAYBACK_CUTOFF_KEY = "cutoff_turn_seq"


class PlaybackOwnershipStamper:
    """把 TTS 音頻幀攜帶的所有權按播放周期復制到 BotStarted/BotStopped。

    上游契約(xbot DLG-029):TTS 側給同一合成 context 的每個
    ``TTSAudioRawFrame`` 蓋 ``audio_ownership_version`` / ``context_id`` /
    ``turn_id``(系統播報無 turn_id)。

    對位原理 —— 到達流周期邊界重建:MediaSender 的播放周期由 sink 隊列
    中的音頻段與 ``TTSStoppedFrame`` 的順序決定,而 sink 順序 == 幀到達
    輸出傳輸(process_frame)的順序。因此在到達流上,「每個 Stopped 邊界
    之後的第一個音頻幀」恰好對應一個播放周期的開端 —— 按此登記每周期
    一條快照,天然覆蓋 vendor 每句 yield 一個 Stopped(同 context 多周期)
    的形態,對齊的是 MediaSender 的真實周期結構,不是每 context 一條的
    臆想契約。

    覆核定下的三條防線:
    - 斷流恢復免疫:pipecat 在 sink 空置 3s 後會補發 Stopped(內部產生,
      不經到達流),恢復時同 context 二次 Started。此時 sink 已排空,
      pending 必為空 —— pending 空的 Started 沿用上一周期身份,不彈錯
      後續快照。
    - 無章佔位:不帶 ownership 版本的音頻同樣開周期,登記空佔位,
      防止旁路音頻偷走後續 context 的身份(當前管線無此類幀,防禦性)。
    - 溢出中毒:pending 超限說明周期事件缺失,丟任何一端都會鏈式錯位
      —— 整隊清空並停止蓋章(退化為 legacy 裸幀),直到下一次打斷重新
      同步。寧可缺身份,不可錯身份。
    """

    _KEYS = ("audio_ownership_version", "context_id", "turn_id", "turn_seq")
    _MAX_PENDING = 8

    def __init__(self) -> None:
        self._pending: deque[dict] = deque()
        self._cycle: Optional[dict] = None
        self._cycle_open = False
        # 到達流上的「音頻段」開關:Stopped 邊界復位;段內後續幀不重複登記。
        self._arrival_run_open = False
        self._poisoned = False

    def _register(self, snapshot: dict) -> None:
        self._arrival_run_open = True
        if self._poisoned:
            return
        if len(self._pending) >= self._MAX_PENDING:
            self._pending.clear()
            self._poisoned = True
            logger.warning(
                "PlaybackOwnershipStamper pending overflow — poisoned until next "
                "interruption (playback events stay unstamped rather than misaligned)"
            )
            return
        self._pending.append(snapshot)

    def observe_audio_frame(self, frame: Frame) -> None:
        if self._arrival_run_open:
            return
        metadata = getattr(frame, "metadata", None) or {}
        if metadata.get("audio_ownership_version") is None:
            self._register({})
            return
        self._register({key: metadata[key] for key in self._KEYS if key in metadata})

    def observe_stop_frame(self) -> None:
        """到達流上的 TTSStoppedFrame = 一個播放周期邊界。

        雙 stop(vendor early-return)在此天然無害:第二個邊界上開關已
        復位,不產生空周期。"""
        self._arrival_run_open = False

    def observe_interruption(self) -> None:
        self._pending.clear()
        self._cycle = None
        self._cycle_open = False
        self._arrival_run_open = False
        self._poisoned = False

    def has_owned_at_or_below(self, cutoff_seq: int) -> bool:
        """sink 已知內容(當前周期或待播快照)是否含序號 <= cutoff 的音頻。

        無主快照(開場白/結束語/佔位)序號不可解析,永遠回 False —— 系統
        播報不因輪次接管被誤清。"""
        snapshots = list(self._pending)
        if self._cycle:
            snapshots.append(self._cycle)
        for snapshot in snapshots:
            seq = _owned_seq(snapshot)
            if seq is not None and seq <= cutoff_seq:
                return True
        return False

    def stamp(self, frame: Frame) -> None:
        if isinstance(frame, BotStartedSpeakingFrame):
            if not self._cycle_open:
                self._cycle_open = True
                if self._pending:
                    self._cycle = self._pending.popleft()
                # pending 空 = 同 context 斷流恢復(3s fallback 補停後同段
                # 音頻續播)—— 沿用上一周期身份;若上一周期本就是無章
                # 佔位,沿用結果仍是不蓋章,自洽。
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # 周期關閉後保留快照:同一事件成對推兩幀(下行+上行),
            # 第二幀仍需蓋章;下一次周期開啟時整體替換。
            self._cycle_open = False
        if self._cycle:
            frame.metadata.update(self._cycle)


# ─── Input Transport ────────────────────────────────────────────────────────


class SipInputTransport(BaseInputTransport):
    """Receives audio + DTMF from SIP call.

    Audio is received from the Rust endpoint via blocking recv (GIL released),
    decoded and resampled in Rust, and pushed as InputAudioRawFrame into the
    Pipecat pipeline. DTMF and call lifecycle events are polled separately.
    """

    def __init__(self, endpoint, session_id: str, transport: "SipTransport",
                 params: Optional[TransportParams] = None,
                 event_queue: Optional[asyncio.Queue] = None, **kwargs):
        if params is None:
            params = TransportParams(
                audio_in_enabled=True,
                audio_in_passthrough=True,
            )
        # Pipecat's BaseInputTransport.start() reads params.audio_in_sample_rate
        # to set self._sample_rate. The Rust endpoint emits frames at its fixed
        # configured rate; if the user didn't specify one, propagate the
        # endpoint's rate so VAD/turn analyzers/audio-filters see matching Hz.
        # If the user specified a mismatched rate, log a warning — the frames
        # will still carry the endpoint's rate in their metadata but downstream
        # processors may miscompute based on the transport's sample_rate.
        _ep_in = endpoint.input_sample_rate
        if params.audio_in_sample_rate is None:
            params.audio_in_sample_rate = _ep_in
        elif params.audio_in_sample_rate != _ep_in:
            logger.warning(
                "SipInputTransport: params.audio_in_sample_rate={} does not match "
                "endpoint.input_sample_rate={}. Rust emits frames at the endpoint rate; "
                "reconfigure the endpoint at construction to match.",
                params.audio_in_sample_rate, _ep_in,
            )
        super().__init__(params, **kwargs)
        self._ep = endpoint
        self._cid = session_id
        self._transport = transport
        self._event_queue = event_queue
        # Started flag — distinct from _paused (the base-class pause flag).
        # _started gates start()/stop() idempotency. The recv loop runs
        # while the tasks exist; pause/resume are handled by the base class
        # via `self._paused` which `push_audio_frame` already gates on.
        self._started = False
        self._recv_task: Optional[asyncio.Task] = None
        self._event_task: Optional[asyncio.Task] = None

    async def start(self, frame: StartFrame):
        if self._started:
            return
        await super().start(frame)
        self._started = True
        await self.set_transport_ready(frame)
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._event_task = asyncio.create_task(self._event_loop())
        await self._transport._call_event_handler("on_client_connected")

    async def stop(self, frame: EndFrame):
        if not self._started:
            await super().stop(frame)
            return
        self._started = False
        await self._cancel_tasks()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        self._started = False
        await self._cancel_tasks()
        await super().cancel(frame)

    # NOTE: pause(StopFrame) is intentionally NOT overridden — we let the
    # base class set `self._paused = True`, which `push_audio_frame()`
    # already gates on (`pipecat/transports/base_input.py:319-326`).
    # The recv loop keeps reading from Rust and producing frames, but the
    # frames are dropped at `push_audio_frame` while paused. On resume the
    # base class sets `self._paused = False` and frames flow again — no
    # need to cancel + recreate tasks. This matches base-class semantics
    # without requiring a custom restart mechanism.

    async def _cancel_tasks(self):
        """Cancel and await background tasks."""
        tasks = [t for t in [self._recv_task, self._event_task] if t and not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._recv_task = None
        self._event_task = None

    async def _recv_loop(self):
        """Receive audio from Rust endpoint via blocking call (GIL released)."""
        loop = asyncio.get_running_loop()
        try:
            while self._started:
                try:
                    result = await loop.run_in_executor(
                        audio_io_executor(), lambda: self._ep.recv_audio_bytes_blocking(self._cid, 20)
                    )
                except Exception:
                    # Session ended (remote BYE removed the call from the
                    # Rust call map). Exit the recv loop cleanly — the
                    # event loop is responsible for firing on_client_disconnected.
                    break
                if result is not None:
                    audio_bytes, sample_rate, num_channels = result
                    # push_audio_frame() drops the frame if base class
                    # `_paused == True`, so no extra check needed here.
                    await self.push_audio_frame(InputAudioRawFrame(
                        audio=bytes(audio_bytes), sample_rate=sample_rate, num_channels=num_channels,
                    ))
        except asyncio.CancelledError:
            raise

    async def _event_loop(self):
        """Poll for DTMF, call state, and lifecycle events.

        If an event_queue is provided (by the server), reads from it.
        Otherwise falls back to polling endpoint directly (standalone usage).
        """
        if self._event_queue:
            await self._event_loop_from_queue()
        else:
            await self._event_loop_from_endpoint()

    async def _event_loop_from_queue(self):
        """Read events from per-session queue (dispatched by server)."""
        while self._started:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.debug("SipInputTransport event_loop error: {}", e)
                break
            try:
                await self._handle_event(event)
            except Exception:
                logger.exception("SipInputTransport handler failed for event %r", event.get("type") if isinstance(event, dict) else event)

    async def _event_loop_from_endpoint(self):
        """Subscribe to GLOBAL_DICT for events on our session id.

        Used when this transport is constructed without a server (no
        ``_event_queue`` provided) — pipecat creates the SipTransport
        directly and we own the endpoint's event flow.
        """
        # Install the shared sink (idempotent; safe if the server in
        # another instance already installed it).
        try:
            self._ep.set_event_sink(_on_event_from_rust)
        except Exception:
            logger.debug("set_event_sink failed (already installed?)", exc_info=True)

        cid = self._cid

        def _is_ours(e: dict) -> bool:
            # Endpoint-level events have no session_id; filter to events
            # for our call. Lifecycle (registered/etc) is irrelevant
            # to InputTransport.
            return e.get("session_id") == cid

        queue = GLOBAL_DICT.subscribe(filter_fn=_is_ours)
        try:
            while self._started:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.debug("SipInputTransport endpoint event_loop error: {}", e)
                    break
                try:
                    await self._handle_event(event)
                except Exception:
                    logger.exception("SipInputTransport handler failed for event %r", event.get("type") if isinstance(event, dict) else event)
        finally:
            GLOBAL_DICT.unsubscribe(queue)

    async def _handle_event(self, event):
        """Process a single event."""
        event_type = event.get("type", "")
        if event_type == "dtmf_received":
            await self.push_frame(InputDTMFFrame(button=KeypadEntry(event["digit"])))
        elif event_type == "call_terminated":
            self._started = False
            await self._transport._call_event_handler("on_client_disconnected")
            await self.push_frame(EndFrame())
        elif event_type == "beep_detected":
            await self._transport._call_event_handler(
                "on_beep_detected",
                event.get("frequency_hz"), event.get("duration_ms"),
            )
        elif event_type == "beep_timeout":
            await self._transport._call_event_handler("on_beep_timeout")
        elif event_type in (
            "audio_capture_complete",
            "audio_playout_complete",
            "audio_buffer_drained",
            "audio_capture_error",
        ):
            # LiveKit-faithful FfiQueue dispatch: every async_id event
            # goes into the shared transport broker. OutputTransport
            # subscribes per-frame inside write_audio_frame and filters
            # to matching async_id.
            self._transport._events.put(event)


# ─── Output Transport ───────────────────────────────────────────────────────


class SipOutputTransport(BaseOutputTransport):
    """Sends audio to SIP call.

    Uses Pipecat's MediaSender infrastructure (via set_transport_ready) for:
    - Audio chunking to transport chunk size
    - BotStartedSpeaking/BotStoppedSpeaking state management
    - Proper interruption handling (task cancellation + restart)

    Audio is sent to the Rust endpoint which handles:
    - RTP packetization and 20ms pacing
    - G.711 codec encoding
    - Jitter buffer, PLC, comfort noise (if enabled)
    """

    def __init__(self, endpoint, session_id: str, transport: "SipTransport",
                 params: Optional[TransportParams] = None, **kwargs):
        if params is None:
            params = TransportParams(
                audio_out_enabled=True,
            )
        # Match Rust endpoint's outbound sample rate. Pipecat's MediaSender
        # chunks/paces audio against `params.audio_out_sample_rate`; if the
        # user leaves it unset, propagate the endpoint rate so the chunking
        # math matches the rate actually sent on the wire.
        _ep_out = endpoint.output_sample_rate
        if params.audio_out_sample_rate is None:
            params.audio_out_sample_rate = _ep_out
        elif params.audio_out_sample_rate != _ep_out:
            logger.warning(
                "SipOutputTransport: params.audio_out_sample_rate={} does not match "
                "endpoint.output_sample_rate={}. Rust resamples input but expects "
                "frames carrying the native sample rate in frame.sample_rate.",
                params.audio_out_sample_rate, _ep_out,
            )
        super().__init__(params, **kwargs)
        self._ep = endpoint
        self._cid = session_id
        self._transport = transport
        self._started = False
        # 播放邊界所有權復制(xbot DLG-029 契約的傳輸側半邊)。
        self._playback_ownership = PlaybackOwnershipStamper()
        # 輪次接管(DLG-030):序號 <= cutoff 的有主音頻失去播放權。單調
        # 遞增,跨打斷保留 —— canonical 序號是通話級的。-1 = 從未收到信號。
        self._playback_cutoff = -1
        self._supersede_dropped: set = set()
        # FfiQueue lives on self._transport — InputTransport pumps,
        # OutputTransport subscribes per-frame. Imported here to avoid
        # circular imports during module-load time.
        from agent_transport._ffi_queue import (
            ASYNC_ID_EVENT_TYPES as _ASYNC_ID_EVENT_TYPES,
            DEFAULT_WAIT_TIMEOUT as _DEFAULT_WAIT_TIMEOUT,
        )
        self._async_id_event_types = _ASYNC_ID_EVENT_TYPES
        self._wait_timeout = _DEFAULT_WAIT_TIMEOUT

    async def start(self, frame: StartFrame):
        if self._started:
            return
        self._started = True
        await super().start(frame)
        self._loop = asyncio.get_running_loop()
        await self.set_transport_ready(frame)

    def _supports_native_dtmf(self) -> bool:
        return True

    async def _write_dtmf_native(self, frame):
        digit = str(frame.button.value)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._ep.send_dtmf(self._cid, digit))

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Send audio frame to SIP call with Rust backpressure.

        LiveKit-faithful subscribe-before-request pattern:
        1. Subscribe to transport.events with a filter narrowing to this
           call's async_id-bearing events.
        2. Call send_audio_async to push the frame and get the async_id.
        3. Await the matching ``audio_capture_complete`` (or
           ``audio_capture_error``) via Queue.wait_for(predicate).
        4. Unsubscribe in finally.

        The subscribe happens BEFORE the FFI call, so even if the Rust
        side emits the completion event synchronously during the push
        (immediate-emit path for below-threshold buffer), it lands in
        our Queue and wait_for finds it.
        """
        cid = self._cid

        def _is_ours(e: dict) -> bool:
            return (
                e.get("type") in self._async_id_event_types
                and e.get("session_id") == cid
            )

        queue = self._transport._events.subscribe(
            loop=asyncio.get_running_loop(),
            filter_fn=_is_ours,
        )
        try:
            try:
                async_id = self._ep.send_audio_async(
                    self._cid,
                    frame.audio,
                    frame.sample_rate,
                    frame.num_channels,
                )
            except Exception as push_err:
                # A rejected push ("buffer full" when the Rust queue is at
                # capacity) used to drop the frame silently — on the wire
                # that shreds speech into zero-padded fragments (measured
                # 2026-07-18: 25-40% zero frames inside TTS windows). The
                # queue drains 20ms per tick, so back off briefly and retry
                # before giving the frame up, and always leave a log trail.
                async_id = None
                for _ in range(3):
                    await asyncio.sleep(0.06)
                    try:
                        async_id = self._ep.send_audio_async(
                            self._cid,
                            frame.audio,
                            frame.sample_rate,
                            frame.num_channels,
                        )
                        break
                    except Exception:
                        continue
                if async_id is None:
                    logger.warning(
                        "write_audio_frame dropped after retries cid={} err={}",
                        cid,
                        push_err,
                    )
                    return False
            try:
                ev = await queue.wait_for(
                    lambda e: e.get("async_id") == async_id,
                    timeout=self._wait_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "write_audio_frame completion timeout cid={} async_id={} waited={}s",
                    cid,
                    async_id,
                    self._wait_timeout,
                )
                return False
        finally:
            self._transport._events.unsubscribe(queue)

        if ev.get("type") == "audio_capture_error" or ev.get("cancelled"):
            # Either a real error (audio_capture_error with non-empty
            # reason) or a clear-induced silent discard (Rust emits
            # AudioCaptureComplete with cancelled=true on clear_buffer).
            # Both mean the frame did NOT reach the wire — surface that
            # to pipecat's MediaSender via the `False` return.
            return False
        return True

    def queued_frames(self) -> int:
        """Number of 20ms audio frames buffered in the Rust outgoing queue."""
        try:
            return self._ep.queued_frames(self._cid)
        except Exception:
            return 0

    async def send_message(self, frame):
        """Send OutputTransportMessageFrame as SIP INFO with JSON body.

        Filters RTVI internal messages (matching should_ignore_frame).
        """
        if isinstance(frame.message, dict) and frame.message.get("label") == "rtvi-ai":
            return
        try:
            msg = json.dumps(frame.message) if not isinstance(frame.message, str) else frame.message
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, lambda: self._ep.send_info(self._cid, "application/json", msg)
            )
        except Exception as e:
            logger.warning("send_message via SIP INFO failed: {}", e)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Clear the local Rust buffer on interruption, then forward to base.

        On SIP, ``clear_buffer`` is *local-only* — it clears the Rust audio
        buffer and resets the resampler, with NO network signaling or
        round-trip. (This is the key difference from the audio_stream
        transport, where clear_buffer additionally sends ``clearAudio`` to
        Plivo; cutting that mid-stream caused ~1.5s of silence on every
        interrupt, so the audio_stream transport deliberately omits it and
        relies on MediaSender task cancellation instead.)

        Because the SIP clear is cheap and never touches the network, doing it
        on barge-in is worth it: it immediately drops up to ~200ms of
        already-buffered TTS instead of letting it play out after the caller
        has interrupted, tightening barge-in latency. ``clear_buffer`` is
        idempotent on terminated sessions (the 0.2.0 Terminated lifecycle
        short-circuits), so no defensive try/except is needed.
        """
        metadata = getattr(frame, "metadata", None) or {}
        if isinstance(frame, InterruptionFrame):
            self._ep.clear_buffer(self._cid)
            # 被沖掉的音頻不會再播:待播所有權快照與 Rust 緩衝同步清空,
            # 否則殘留身份會錯配到下一次播放事件。
            self._playback_ownership.observe_interruption()
            self._supersede_dropped.clear()
        elif metadata.get(_PLAYBACK_CONTROL_VERSION_KEY) is not None:
            self._apply_supersede(metadata)
        elif (
            isinstance(frame, TTSAudioRawFrame)
            and direction == FrameDirection.DOWNSTREAM
        ):
            if self._is_superseded_audio(metadata):
                # 失權輪次音頻:不入 MediaSender、不登記所有權、不播放。
                return
            self._playback_ownership.observe_audio_frame(frame)
        elif (
            isinstance(frame, TTSStoppedFrame)
            and direction == FrameDirection.DOWNSTREAM
        ):
            self._playback_ownership.observe_stop_frame()
        await super().process_frame(frame, direction)

    def _apply_supersede(self, control: dict) -> None:
        """消費輪次接管信號(xbot DLG-030 契約的傳輸側半邊)。

        按 metadata 契約消費(playback_control_version / cutoff_turn_seq),
        不依賴跨倉類同一性。僅當 sink 已知內容確含失權輪次的有主音頻時
        才清 Rust 緩衝並整體重置所有權狀態 —— 清後仍從 MediaSender 滲出
        的無身份殘餘退化為無主播放並被計量排除(缺身份勝於錯身份);
        無主系統播報(開場白/結束語)不受接管影響。cutoff 單調:遲到或
        重復的信號是空操作。發射側每輪接管必發,選擇性由本側保證。"""
        try:
            cutoff = int(control.get(_PLAYBACK_CUTOFF_KEY))
        except (TypeError, ValueError):
            return
        if cutoff <= self._playback_cutoff:
            return
        self._playback_cutoff = cutoff
        if self._playback_ownership.has_owned_at_or_below(cutoff):
            self._ep.clear_buffer(self._cid)
            self._playback_ownership.observe_interruption()
            logger.info(
                "[Supersede] stale playback cleared cid={} cutoff_turn_seq={}",
                self._cid, cutoff,
            )

    def _is_superseded_audio(self, metadata: dict) -> bool:
        if self._playback_cutoff < 0:
            return False
        if metadata.get("audio_ownership_version") is None:
            return False
        seq = _owned_seq(metadata)
        if seq is None or seq > self._playback_cutoff:
            return False
        owner = str(metadata.get("turn_id") or f"seq-{seq}")
        if owner not in self._supersede_dropped:
            self._supersede_dropped.add(owner)
            logger.info(
                "[Supersede] dropping stale audio cid={} turn={} cutoff_turn_seq={}",
                self._cid, owner, self._playback_cutoff,
            )
        return True

    async def push_frame(
        self,
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ):
        """MediaSender 經此推送成對的 BotStarted/BotStopped(裸幀、無
        metadata)—— 在唯一出口把當前播放周期的所有權復制上去,下游
        (xbot latency tap)按 turn_id/context_id/audio_ownership_version
        直接消費,無需任何順序推斷。"""
        if isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
            self._playback_ownership.stamp(frame)
        await super().push_frame(frame, direction)

    async def stop(self, frame: EndFrame):
        # ``hangup`` is idempotent on terminated sessions; on Active it
        # tears down the dialog + RTP. No try/except needed.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._ep.hangup(self._cid))
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._ep.hangup(self._cid))
        await super().cancel(frame)


# ─── Composite Transport ────────────────────────────────────────────────────


class SipTransport(BaseTransport):
    """Pipecat transport for SIP calls via agent-transport.

    This is the main entry point. It creates lazily-initialized input and
    output transport processors that are wired into a Pipecat pipeline.

    Architecture:
        SipEndpoint (Rust) handles SIP signaling, RTP audio, codec encoding,
        and 20ms pacing. This transport bridges Pipecat frames to the Rust endpoint.

    Event handlers (register via @transport.event_handler("name")):
        on_client_connected  — SIP call answered, media active
        on_client_disconnected — call terminated (BYE or hangup)
        on_beep_detected — voicemail beep detected
        on_beep_timeout — beep detection timed out

    Session metadata (from SIP call session):
        transport.session_id    — internal session ID
        transport.call_uuid     — SIP call UUID
        transport.remote_uri    — remote SIP URI
        transport.direction     — "Inbound" or "Outbound"
        transport.extra_headers — custom SIP headers

    Audio control:
        mute() / unmute()           — mute/unmute outgoing audio
        pause_playback() / resume_playback() — pause/resume RTP send loop
        clear_buffer()              — clear queued audio
        send_background_audio()     — mix background audio with agent voice
        start_recording() / stop_recording() — WAV stereo call recording
        flush() / wait_for_playout() — track playback completion

    SIP-specific:
        transfer() / transfer_attended() — call transfer
        hold() / unhold() — SIP hold (re-INVITE)
        reject() — reject incoming call
        detect_beep() / cancel_beep_detection() — voicemail detection
        send_dtmf() — send DTMF digits
        send_info() — send SIP INFO message
    """

    def __init__(
        self,
        endpoint,
        session_id: str,
        *,
        name: Optional[str] = None,
        params: Optional[TransportParams] = None,
        session_data: Optional[Dict[str, Any]] = None,
        _event_queue: Optional[asyncio.Queue] = None,
        **kwargs,
    ):
        super().__init__(name=name or "SipTransport", **kwargs)
        self._ep = endpoint
        self._cid = session_id
        self._params = params
        self._session_data = session_data or {}
        self._event_queue = _event_queue
        # Shared FfiQueue for this transport — InputTransport's event
        # loop puts async_id events into it; OutputTransport subscribes
        # per-frame. LiveKit-faithful subscribe-before-request pattern.
        from agent_transport._ffi_queue import FfiQueue
        self._events: FfiQueue = FfiQueue()
        self._input: Optional[SipInputTransport] = None
        self._output: Optional[SipOutputTransport] = None

        # Register Pipecat event handlers
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_beep_detected")
        self._register_event_handler("on_beep_timeout")

    # ── Pipeline processors ──────────────────────────────────────────────

    def input(self) -> SipInputTransport:
        if self._input is None:
            self._input = SipInputTransport(
                self._ep, self._cid, transport=self,
                params=self._params, event_queue=self._event_queue,
                name=f"{self._name}-input",
            )
        return self._input

    def output(self) -> SipOutputTransport:
        if self._output is None:
            self._output = SipOutputTransport(
                self._ep, self._cid, transport=self,
                params=self._params, name=f"{self._name}-output",
            )
        return self._output

    # ── Session metadata ─────────────────────────────────────────────────

    @property
    def session_id(self) -> str:
        """Internal session ID used to address this session."""
        return self._cid

    @property
    def call_uuid(self) -> str:
        """SIP call UUID."""
        return self._session_data.get("call_uuid", self._cid)

    @property
    def remote_uri(self) -> str:
        """Remote SIP URI (e.g., sip:+1234567890@provider.com)."""
        return self._session_data.get("remote_uri", "")

    @property
    def direction(self) -> str:
        """Call direction: 'Inbound' or 'Outbound'."""
        return self._session_data.get("direction", "")

    @property
    def extra_headers(self) -> Dict[str, str]:
        """Custom SIP headers from the INVITE."""
        return self._session_data.get("extra_headers", {})

    # ── Audio control ────────────────────────────────────────────────────

    def mute(self) -> None:
        """Mute outgoing audio. RTP sends silence, queue preserved."""
        self._ep.mute(self._cid)

    def unmute(self) -> None:
        """Unmute outgoing audio."""
        self._ep.unmute(self._cid)

    def pause_playback(self) -> None:
        """Pause the RTP send loop. Queue accumulates, nothing sent."""
        self._ep.pause(self._cid)

    def resume_playback(self) -> None:
        """Resume the RTP send loop."""
        self._ep.resume(self._cid)

    def clear_buffer(self) -> None:
        """Clear local audio buffer (fires pending completion callbacks)."""
        self._ep.clear_buffer(self._cid)

    # ── Background audio ─────────────────────────────────────────────────

    def send_background_audio(self, audio: bytes, sample_rate: int, num_channels: int) -> None:
        """Send background audio to be mixed with agent voice in Rust's RTP send loop."""
        self._ep.send_background_audio(self._cid, audio, sample_rate, num_channels)

    # ── Flush / playout tracking ─────────────────────────────────────────

    def flush(self) -> None:
        """Mark current playback segment complete."""
        self._ep.flush(self._cid)

    async def wait_for_playout(self, timeout_ms: int = 5000) -> bool:
        """Wait for queued audio to finish playing.

        Returns True if playout completed, False if timed out.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._ep.wait_for_playout(self._cid, timeout_ms)
        )

    # ── Recording ────────────────────────────────────────────────────────

    def start_recording(self, path: str, stereo: bool = True) -> None:
        """Start recording the call to a WAV file.

        Args:
            path: Output file path (.wav).
            stereo: If True, record stereo (L=user, R=agent). Otherwise mono.
        """
        self._ep.start_recording(self._cid, path, stereo)

    def stop_recording(self) -> None:
        """Stop recording the call."""
        self._ep.stop_recording(self._cid)

    # ── DTMF ─────────────────────────────────────────────────────────────

    def send_dtmf(self, digits: str, method: str = "rfc2833") -> None:
        """Send DTMF digits.

        Args:
            digits: One or more DTMF digits (0-9, *, #, A-D).
            method: 'rfc2833' (in-band RTP) or 'sip_info' (SIP INFO).
        """
        self._ep.send_dtmf(self._cid, digits, method)

    # ── SIP call control ─────────────────────────────────────────────────

    def transfer(self, dest_uri: str) -> None:
        """Blind transfer — transfer call to another SIP URI."""
        self._ep.transfer(self._cid, dest_uri)

    def transfer_attended(self, target_session_id: str) -> None:
        """Attended transfer — transfer to an existing call."""
        self._ep.transfer_attended(self._cid, target_session_id)

    def hold(self) -> None:
        """Put the call on hold (SIP re-INVITE with a=sendonly)."""
        self._ep.hold(self._cid)

    def unhold(self) -> None:
        """Take the call off hold (SIP re-INVITE with a=sendrecv)."""
        self._ep.unhold(self._cid)

    def reject(self, code: int = 486) -> None:
        """Reject an incoming call with a SIP status code."""
        self._ep.reject(self._cid, code)

    def send_info(self, content_type: str = "application/json", body: str = "") -> None:
        """Send a SIP INFO message."""
        self._ep.send_info(self._cid, content_type, body)

    # ── Beep detection ───────────────────────────────────────────────────

    def detect_beep(self, timeout_ms: int = 30000, min_duration_ms: int = 80,
                    max_duration_ms: int = 5000) -> None:
        """Start voicemail beep detection.

        Fires on_beep_detected or on_beep_timeout event when done.
        """
        self._ep.detect_beep(self._cid, timeout_ms, min_duration_ms, max_duration_ms)

    def cancel_beep_detection(self) -> None:
        """Cancel ongoing beep detection."""
        self._ep.cancel_beep_detection(self._cid)

    # ── Raw message ──────────────────────────────────────────────────────

    def send_raw_message(self, content_type: str, body: str) -> None:
        """Send a raw SIP INFO message."""
        self._ep.send_info(self._cid, content_type, body)
