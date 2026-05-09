"""
Audio utilities — normalize, resample, export WAV/MP3, merge chunks.

All functions accept and return numpy float32 arrays unless stated otherwise.
Sample rate is always passed explicitly; these functions are stateless.
"""

import shutil
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# normalize_audio
# ---------------------------------------------------------------------------

def normalize_audio(audio_np: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio loudness to *target_db* (RMS-based).

    Args:
        audio_np:  Input audio as float32 array, values in [-1, 1].
        target_db: Target RMS level in dBFS. Default -20.0 dBFS.

    Returns:
        Gain-adjusted float32 array, clipped to [-1, 1].
    """
    audio = np.asarray(audio_np, dtype=np.float32)
    rms = float(np.sqrt(np.mean(audio ** 2)))

    if rms < 1e-9:
        warnings.warn("normalize_audio: input is silent (RMS ≈ 0), returning as-is.")
        return audio

    target_rms = 10 ** (target_db / 20.0)
    gain = target_rms / rms
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# resample_audio
# ---------------------------------------------------------------------------

def resample_audio(
    audio_np: np.ndarray,
    from_sr: int,
    to_sr: int,
) -> np.ndarray:
    """
    Resample audio from *from_sr* to *to_sr* Hz.

    Uses scipy.signal.resample_poly when available (higher quality),
    falls back to numpy linear interpolation (adequate for speech).

    Args:
        audio_np: 1-D float32 array.
        from_sr:  Source sample rate.
        to_sr:    Target sample rate.

    Returns:
        Resampled float32 array.
    """
    audio = np.asarray(audio_np, dtype=np.float32)

    if from_sr == to_sr:
        return audio

    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(to_sr, from_sr)
        return resample_poly(audio, to_sr // g, from_sr // g).astype(np.float32)
    except ImportError:
        pass

    # numpy linear interpolation fallback
    old_len = len(audio)
    new_len = int(round(old_len * to_sr / from_sr))
    x_old = np.linspace(0.0, 1.0, old_len)
    x_new = np.linspace(0.0, 1.0, new_len)
    return np.interp(x_new, x_old, audio).astype(np.float32)


# ---------------------------------------------------------------------------
# export_wav
# ---------------------------------------------------------------------------

def export_wav(
    audio_np: np.ndarray,
    sample_rate: int,
    output_path: str | Path,
) -> Path:
    """
    Export audio to a WAV file using soundfile.

    Args:
        audio_np:    1-D float32 array.
        sample_rate: Sample rate in Hz.
        output_path: Destination file path (extension must be .wav).

    Returns:
        Resolved Path of the written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio_np, dtype=np.float32)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")
    return path.resolve()


# ---------------------------------------------------------------------------
# export_mp3
# ---------------------------------------------------------------------------

def export_mp3(
    audio_np: np.ndarray,
    sample_rate: int,
    output_path: str | Path,
    bitrate: str = "192k",
) -> Path:
    """
    Export audio to an MP3 file via pydub + ffmpeg.

    Falls back to WAV (with a warning) when ffmpeg is not installed.

    Args:
        audio_np:    1-D float32 array.
        sample_rate: Sample rate in Hz.
        output_path: Destination file path (.mp3 recommended).
        bitrate:     MP3 bitrate string, e.g. "192k". Default "192k".

    Returns:
        Resolved Path of the written file (may be .wav if ffmpeg missing).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not shutil.which("ffmpeg"):
        warnings.warn(
            "ffmpeg not found — exporting as WAV instead of MP3. "
            "Install ffmpeg to enable MP3 export.",
            stacklevel=2,
        )
        wav_path = path.with_suffix(".wav")
        return export_wav(audio_np, sample_rate, wav_path)

    from pydub import AudioSegment

    audio = np.asarray(audio_np, dtype=np.float32)
    # pydub expects int16 PCM bytes
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    segment = AudioSegment(
        pcm,
        frame_rate=sample_rate,
        sample_width=2,   # int16 = 2 bytes
        channels=1,
    )
    segment.export(str(path), format="mp3", bitrate=bitrate)
    return path.resolve()


# ---------------------------------------------------------------------------
# merge_audio_chunks
# ---------------------------------------------------------------------------

def merge_audio_chunks(
    chunks: list[np.ndarray],
    silence_ms: int = 300,
    sample_rate: int = 24_000,
    chunk_sample_rates: list[int] | None = None,
) -> np.ndarray:
    """
    Concatenate multiple audio chunks with silence padding between them.

    When chunks have different sample rates (e.g. EN/VI at 24 kHz mixed with
    JA at 44.1 kHz), pass *chunk_sample_rates* to resample everything to
    *sample_rate* before merging.

    Args:
        chunks:             List of 1-D float32 arrays.
        silence_ms:         Milliseconds of silence inserted between chunks.
        sample_rate:        Target output sample rate.
        chunk_sample_rates: Per-chunk sample rate. If None, assumes all chunks
                            are already at *sample_rate*.

    Returns:
        Merged float32 array at *sample_rate*.
    """
    if not chunks:
        return np.array([], dtype=np.float32)

    silence_samples = int(sample_rate * silence_ms / 1000)
    silence = np.zeros(silence_samples, dtype=np.float32)

    parts: list[np.ndarray] = []
    for i, chunk in enumerate(chunks):
        chunk = np.asarray(chunk, dtype=np.float32)

        # Resample if source SR differs from target SR
        if chunk_sample_rates is not None:
            src_sr = chunk_sample_rates[i]
            if src_sr != sample_rate:
                chunk = resample_audio(chunk, src_sr, sample_rate)

        parts.append(chunk)
        if i < len(chunks) - 1:
            parts.append(silence)

    return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------------------
# get_duration
# ---------------------------------------------------------------------------

def get_duration(audio_np: np.ndarray, sample_rate: int) -> float:
    """Return the duration of *audio_np* in seconds."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    return len(audio_np) / sample_rate


# ---------------------------------------------------------------------------
# validate_output_dir
# ---------------------------------------------------------------------------

def validate_output_dir(path: str | Path) -> Path:
    """
    Ensure *path* exists as a writable directory.

    Creates the directory (and parents) if it does not yet exist.

    Args:
        path: Directory path string or Path object.

    Returns:
        Resolved Path object.

    Raises:
        PermissionError: If the directory cannot be created or written to.
        NotADirectoryError: If *path* exists but is a file.
    """
    p = Path(path).resolve()

    if p.exists() and not p.is_dir():
        raise NotADirectoryError(f"Path exists but is a file, not a directory: {p}")

    try:
        p.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise PermissionError(f"Cannot create output directory (permission denied): {p}")

    # Verify we can actually write into the directory
    probe = p / ".write_probe"
    try:
        probe.touch()
        probe.unlink()
    except OSError as exc:
        raise PermissionError(
            f"Output directory is not writable: {p}\n{exc}"
        ) from exc

    return p
