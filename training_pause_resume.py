from __future__ import annotations

import contextlib
import json
import os
import re
import signal
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")
DEFAULT_STATE_FILENAME = "pause_resume_state.json"

MODEL_STATE_FILES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "pytorch_model.bin",
    "model.safetensors",
)
OPTIMIZER_STATE_FILES = (
    "optimizer.pt",
    "optimizer.bin",
    "optimizer_state.pt",
    "optimizer_state.bin",
)
SCHEDULER_STATE_FILES = (
    "scheduler.pt",
    "scheduler.bin",
    "scheduler_state.pt",
    "scheduler_state.bin",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _abs_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def _maybe_abs_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    return _abs_path(path_str)


def _path_to_str(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    return str(path.resolve())


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.tmp"
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, path)


def read_state_file(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        return None, None

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except Exception as exc:
        return None, f"Could not read pause/resume state file {path}: {exc}"


def checkpoint_step_from_name(name: str) -> Optional[int]:
    match = CHECKPOINT_DIR_RE.match(name)
    if match is None:
        return None
    return int(match.group(1))


@dataclass(frozen=True)
class CheckpointInspection:
    path: Path
    step: int
    is_complete: bool
    missing: Tuple[str, ...]


def inspect_checkpoint(checkpoint_dir: Path) -> CheckpointInspection:
    step = checkpoint_step_from_name(checkpoint_dir.name)
    if step is None:
        raise ValueError(f"Expected a Trainer checkpoint directory named checkpoint-<step>, got: {checkpoint_dir}")

    missing: List[str] = []
    if not checkpoint_dir.is_dir():
        missing.append("checkpoint directory")
        return CheckpointInspection(
            path=checkpoint_dir.resolve(),
            step=step,
            is_complete=False,
            missing=tuple(missing),
        )

    file_names = {child.name for child in checkpoint_dir.iterdir() if child.is_file()}
    if "trainer_state.json" not in file_names:
        missing.append("trainer_state.json")
    if not any(name in file_names for name in MODEL_STATE_FILES):
        missing.append("model weights")
    if not any(name in file_names for name in OPTIMIZER_STATE_FILES):
        missing.append("optimizer state")
    if not any(name in file_names for name in SCHEDULER_STATE_FILES):
        missing.append("scheduler state")
    if not any(name.startswith("rng_state") and name.endswith(".pth") for name in file_names):
        missing.append("rng state")

    return CheckpointInspection(
        path=checkpoint_dir.resolve(),
        step=step,
        is_complete=not missing,
        missing=tuple(missing),
    )


def discover_checkpoints(output_dir: str | Path) -> List[CheckpointInspection]:
    output_path = Path(output_dir)
    if not output_path.exists():
        return []

    inspections: List[CheckpointInspection] = []
    for child in output_path.iterdir():
        if not child.is_dir():
            continue
        if checkpoint_step_from_name(child.name) is None:
            continue
        inspections.append(inspect_checkpoint(child))

    inspections.sort(key=lambda item: item.step, reverse=True)
    return inspections


def find_latest_complete_checkpoint(output_dir: str | Path) -> Optional[CheckpointInspection]:
    for inspection in discover_checkpoints(output_dir):
        if inspection.is_complete:
            return inspection
    return None


def resolve_resume_checkpoint(
    *,
    output_dir: str,
    resume_from_checkpoint: Optional[str],
    state_path: Optional[str] = None,
) -> Tuple[Optional[str], List[str]]:
    if not resume_from_checkpoint:
        return None, []

    warnings: List[str] = []
    resume_value = resume_from_checkpoint.strip()
    if not resume_value:
        return None, []

    if resume_value.lower() != "last":
        checkpoint_path = _abs_path(resume_value)
        if not checkpoint_path.exists():
            raise RuntimeError(f"Resume checkpoint does not exist: {checkpoint_path}")
        inspection = inspect_checkpoint(checkpoint_path)
        if not inspection.is_complete:
            missing = ", ".join(inspection.missing)
            raise RuntimeError(f"Refusing to resume from incomplete checkpoint {checkpoint_path} (missing: {missing}).")
        return str(inspection.path), warnings

    inspections = discover_checkpoints(output_dir)
    latest_complete = next((item for item in inspections if item.is_complete), None)
    if latest_complete is None:
        raise RuntimeError(
            f"--resume_from_checkpoint last was requested, but no complete Trainer checkpoint was found in {Path(output_dir).resolve()}."
        )

    for inspection in inspections:
        if inspection.step <= latest_complete.step:
            break
        if not inspection.is_complete:
            missing = ", ".join(inspection.missing)
            warnings.append(
                f"[Pause/Resume][WARN] Skipping incomplete checkpoint {inspection.path} (missing: {missing})."
            )

    if state_path:
        state_data, state_error = read_state_file(Path(state_path))
        if state_error:
            warnings.append(f"[Pause/Resume][WARN] {state_error}")
        elif state_data and state_data.get("latest_checkpoint"):
            state_checkpoint = _maybe_abs_path(state_data.get("latest_checkpoint"))
            if state_checkpoint is not None and state_checkpoint != latest_complete.path:
                warnings.append(
                    "[Pause/Resume][WARN] State metadata points to "
                    f"{state_checkpoint}, but the latest valid on-disk checkpoint is {latest_complete.path}. "
                    "Resuming from the on-disk checkpoint."
                )

    return str(latest_complete.path), warnings


class PauseResumeManager:
    def __init__(
        self,
        *,
        output_dir: str,
        pause_request_path: Optional[str] = None,
        pause_on_interrupt: bool = False,
    ) -> None:
        self.output_dir = _abs_path(output_dir)
        self.pause_request_path = _maybe_abs_path(pause_request_path)
        self.pause_on_interrupt = pause_on_interrupt
        self.state_path = self.output_dir / DEFAULT_STATE_FILENAME
        self.resume_checkpoint: Optional[Path] = None

        self._pause_request_reason: Optional[str] = None
        self._pause_request_seen_at: Optional[str] = None
        self._pause_armed_step: Optional[int] = None
        self._pause_armed_epoch: Optional[float] = None
        self._pause_armed_via: Optional[str] = None

    @property
    def pause_armed(self) -> bool:
        return self._pause_armed_step is not None

    def describe_startup(self) -> List[str]:
        lines = [f"[Pause/Resume] State file: {self.state_path}"]
        if self.pause_request_path is not None:
            lines.append(
                f"[Pause/Resume] Create {self.pause_request_path} while training is running to request a safe pause."
            )
            if self.pause_request_path.exists():
                lines.append(
                    f"[Pause/Resume][WARN] Pause request file already exists: {self.pause_request_path}. "
                    "Training will pause at the next safe checkpoint boundary unless that file is removed first."
                )
        if self.pause_on_interrupt:
            lines.append(
                "[Pause/Resume] Ctrl+C requests a safe pause instead of stopping immediately."
            )
        if self.resume_checkpoint is not None:
            lines.append(f"[Pause/Resume] Resuming from checkpoint: {self.resume_checkpoint}")
        return lines

    def request_pause(self, reason: str) -> None:
        if self._pause_request_reason is None:
            self._pause_request_reason = reason
            self._pause_request_seen_at = _utc_now_iso()

    def pause_requested(self) -> bool:
        if self._pause_request_reason is not None:
            return True

        if self.pause_request_path is not None and self.pause_request_path.exists():
            self.request_pause(f"file:{self.pause_request_path}")
            return True

        return False

    def write_running_state(self) -> None:
        payload = {
            "version": 1,
            "status": "running",
            "output_dir": _path_to_str(self.output_dir),
            "pause_on_interrupt": self.pause_on_interrupt,
            "pause_request_path": _path_to_str(self.pause_request_path),
            "pid": os.getpid(),
            "resume_checkpoint": _path_to_str(self.resume_checkpoint),
            "updated_at": _utc_now_iso(),
        }
        write_json_atomic(self.state_path, payload)

    def arm_pause(self, *, global_step: int, epoch: Optional[float], via: str) -> None:
        if self.pause_armed:
            return

        self._pause_armed_step = int(global_step)
        self._pause_armed_epoch = epoch
        self._pause_armed_via = via

        reason = self._pause_request_reason or via
        print(
            "[Pause/Resume] Pause requested via "
            f"{reason}. Saving a full checkpoint at global step {self._pause_armed_step} and then stopping."
        )

    def finalize_pause(self, trainer_state: Any) -> Path:
        if not self.pause_armed:
            raise RuntimeError("Cannot finalize pause because no pause request was armed.")

        inspections = discover_checkpoints(self.output_dir)
        exact_match = next(
            (
                item
                for item in inspections
                if item.step == self._pause_armed_step and item.is_complete
            ),
            None,
        )
        if exact_match is None:
            discovered = ", ".join(
                f"{item.path.name} ({'complete' if item.is_complete else 'incomplete'})"
                for item in inspections
            ) or "<none>"
            raise RuntimeError(
                "Pause was requested, but no complete checkpoint was produced for "
                f"global step {self._pause_armed_step}. Discovered checkpoints: {discovered}"
            )

        self._remove_pause_request_file()

        payload = {
            "version": 1,
            "status": "paused",
            "epoch": _safe_float(getattr(trainer_state, "epoch", None)),
            "global_step": int(getattr(trainer_state, "global_step", 0)),
            "latest_checkpoint": _path_to_str(exact_match.path),
            "output_dir": _path_to_str(self.output_dir),
            "pause_armed_via": self._pause_armed_via,
            "pause_on_interrupt": self.pause_on_interrupt,
            "pause_request_path": _path_to_str(self.pause_request_path),
            "pause_request_reason": self._pause_request_reason,
            "pause_requested_at": self._pause_request_seen_at,
            "pid": os.getpid(),
            "resumed_from": _path_to_str(self.resume_checkpoint),
            "updated_at": _utc_now_iso(),
        }
        write_json_atomic(self.state_path, payload)
        return exact_match.path

    def finalize_completion(self, trainer_state: Any) -> None:
        latest_checkpoint = find_latest_complete_checkpoint(self.output_dir)
        self._remove_pause_request_file()

        payload = {
            "version": 1,
            "status": "completed",
            "epoch": _safe_float(getattr(trainer_state, "epoch", None)),
            "global_step": int(getattr(trainer_state, "global_step", 0)),
            "latest_checkpoint": _path_to_str(latest_checkpoint.path if latest_checkpoint else None),
            "output_dir": _path_to_str(self.output_dir),
            "pause_on_interrupt": self.pause_on_interrupt,
            "pause_request_path": _path_to_str(self.pause_request_path),
            "pid": os.getpid(),
            "resumed_from": _path_to_str(self.resume_checkpoint),
            "updated_at": _utc_now_iso(),
        }
        write_json_atomic(self.state_path, payload)

    def _remove_pause_request_file(self) -> None:
        if self.pause_request_path is None:
            return
        try:
            if self.pause_request_path.exists():
                self.pause_request_path.unlink()
        except OSError as exc:
            print(
                f"[Pause/Resume][WARN] Could not remove pause request file {self.pause_request_path}: {exc}"
            )

    @contextlib.contextmanager
    def signal_handlers(self) -> Iterator[None]:
        if not self.pause_on_interrupt:
            yield
            return

        handlers: Dict[signal.Signals, Any] = {}
        supported_signals: List[signal.Signals] = [signal.SIGINT]
        sigterm = getattr(signal, "SIGTERM", None)
        if sigterm is not None:
            supported_signals.append(sigterm)

        def _handle_signal(signum: int, _frame: Any) -> None:
            try:
                signal_name = signal.Signals(signum).name
            except ValueError:
                signal_name = str(signum)

            already_requested = self._pause_request_reason is not None
            self.request_pause(f"signal:{signal_name}")
            if already_requested:
                print(
                    f"[Pause/Resume] {signal_name} received again. A safe pause is already pending."
                )
            else:
                print(
                    f"[Pause/Resume] {signal_name} received. Training will pause after the current optimizer step."
                )

        for sig in supported_signals:
            try:
                handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, _handle_signal)
            except (OSError, RuntimeError, ValueError):
                continue

        try:
            yield
        finally:
            for sig, handler in handlers.items():
                try:
                    signal.signal(sig, handler)
                except (OSError, RuntimeError, ValueError):
                    continue


def create_pause_resume_callback(manager: PauseResumeManager):
    from transformers import TrainerCallback

    class PauseResumeCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            manager.write_running_state()
            return control

        def _maybe_arm_pause(self, state, control, source_event: str):
            if manager.pause_armed:
                return control
            if not manager.pause_requested():
                return control

            manager.arm_pause(
                global_step=int(getattr(state, "global_step", 0)),
                epoch=_safe_float(getattr(state, "epoch", None)),
                via=source_event,
            )
            control.should_save = True
            control.should_training_stop = True
            return control

        def on_step_end(self, args, state, control, **kwargs):
            return self._maybe_arm_pause(state, control, "step_end")

        def on_epoch_end(self, args, state, control, **kwargs):
            return self._maybe_arm_pause(state, control, "epoch_end")

    return PauseResumeCallback()
