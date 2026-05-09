import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.audio_utils import (
    normalize_audio,
    resample_audio,
    export_wav,
    export_mp3,
    merge_audio_chunks,
    get_duration,
    validate_output_dir,
)

SR_24 = 24_000
SR_44 = 44_100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine(freq: float = 440.0, sr: int = SR_24, duration: float = 1.0,
          amplitude: float = 0.1) -> np.ndarray:
    """Return a float32 sine wave."""
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))


# ---------------------------------------------------------------------------
# normalize_audio
# ---------------------------------------------------------------------------

class TestNormalizeAudio(unittest.TestCase):

    def test_returns_float32(self):
        out = normalize_audio(_sine())
        self.assertEqual(out.dtype, np.float32)

    def test_rms_reaches_target_db(self):
        for target_db in (-30.0, -20.0, -10.0, -6.0, -3.0):
            with self.subTest(target_db=target_db):
                out = normalize_audio(_sine(amplitude=0.05), target_db=target_db)
                target_rms = 10 ** (target_db / 20.0)
                self.assertAlmostEqual(_rms(out), target_rms, places=3)

    def test_output_clipped_to_minus_one_plus_one(self):
        # Loud input forced to near-clip target should be clipped, not exceed ±1
        out = normalize_audio(_sine(amplitude=0.001), target_db=0.0)
        self.assertLessEqual(float(np.max(np.abs(out))), 1.0)

    def test_silent_input_returns_unchanged_and_warns(self):
        silent = np.zeros(1000, dtype=np.float32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = normalize_audio(silent)
        self.assertEqual(len(out), 1000)
        self.assertTrue(any("silent" in str(x.message).lower() for x in w),
                        "Expected a warning about silent audio")

    def test_preserves_array_length(self):
        tone = _sine(duration=2.0)
        out = normalize_audio(tone, target_db=-20.0)
        self.assertEqual(len(out), len(tone))

    def test_accepts_non_float32_input(self):
        tone_f64 = _sine().astype(np.float64)
        out = normalize_audio(tone_f64)
        self.assertEqual(out.dtype, np.float32)

    def test_default_target_is_minus_20db(self):
        tone = _sine(amplitude=0.5)
        out = normalize_audio(tone)
        target_rms = 10 ** (-20.0 / 20.0)
        self.assertAlmostEqual(_rms(out), target_rms, places=3)


# ---------------------------------------------------------------------------
# resample_audio
# ---------------------------------------------------------------------------

class TestResampleAudio(unittest.TestCase):

    def test_returns_float32(self):
        out = resample_audio(_sine(), from_sr=SR_24, to_sr=SR_44)
        self.assertEqual(out.dtype, np.float32)

    def test_no_op_when_same_rate(self):
        tone = _sine()
        out = resample_audio(tone, from_sr=SR_24, to_sr=SR_24)
        np.testing.assert_array_equal(tone, out)

    def test_upsample_length_24k_to_44k(self):
        tone = _sine(sr=SR_24, duration=1.0)
        out = resample_audio(tone, from_sr=SR_24, to_sr=SR_44)
        expected = int(round(len(tone) * SR_44 / SR_24))
        self.assertAlmostEqual(len(out), expected, delta=1)

    def test_downsample_length_44k_to_24k(self):
        tone = _sine(sr=SR_44, duration=1.0)
        out = resample_audio(tone, from_sr=SR_44, to_sr=SR_24)
        expected = int(round(len(tone) * SR_24 / SR_44))
        self.assertAlmostEqual(len(out), expected, delta=1)

    def test_roundtrip_preserves_duration(self):
        tone = _sine(sr=SR_24, duration=1.0)
        up = resample_audio(tone, from_sr=SR_24, to_sr=SR_44)
        down = resample_audio(up, from_sr=SR_44, to_sr=SR_24)
        self.assertAlmostEqual(len(down), len(tone), delta=2)

    def test_upsample_duration_preserved(self):
        tone = _sine(sr=SR_24, duration=2.0)
        out = resample_audio(tone, from_sr=SR_24, to_sr=SR_44)
        dur_before = len(tone) / SR_24
        dur_after  = len(out) / SR_44
        self.assertAlmostEqual(dur_before, dur_after, places=2)


# ---------------------------------------------------------------------------
# export_wav
# ---------------------------------------------------------------------------

class TestExportWav(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_returns_path_object(self):
        path = export_wav(_sine(), SR_24, self.tmp / "out.wav")
        self.assertIsInstance(path, Path)

    def test_file_exists_after_export(self):
        path = export_wav(_sine(), SR_24, self.tmp / "out.wav")
        self.assertTrue(path.exists())

    def test_file_is_readable_by_soundfile(self):
        import soundfile as sf
        tone = _sine(duration=1.0)
        path = export_wav(tone, SR_24, self.tmp / "out.wav")
        data, sr = sf.read(str(path))
        self.assertEqual(sr, SR_24)
        self.assertGreater(len(data), 0)

    def test_sample_rate_preserved(self):
        import soundfile as sf
        path = export_wav(_sine(sr=SR_44, duration=0.5), SR_44, self.tmp / "out44.wav")
        _, sr = sf.read(str(path))
        self.assertEqual(sr, SR_44)

    def test_duration_preserved(self):
        import soundfile as sf
        tone = _sine(duration=1.5)
        path = export_wav(tone, SR_24, self.tmp / "out.wav")
        data, sr = sf.read(str(path))
        self.assertAlmostEqual(len(data) / sr, 1.5, places=1)

    def test_creates_parent_dirs(self):
        deep = self.tmp / "a" / "b" / "c" / "out.wav"
        path = export_wav(_sine(), SR_24, deep)
        self.assertTrue(path.exists())


# ---------------------------------------------------------------------------
# export_mp3
# ---------------------------------------------------------------------------

class TestExportMp3(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_returns_path_object(self):
        path = export_mp3(_sine(), SR_24, self.tmp / "out.mp3")
        self.assertIsInstance(path, Path)

    def test_file_exists_after_export(self):
        path = export_mp3(_sine(), SR_24, self.tmp / "out.mp3")
        self.assertTrue(path.exists())

    def test_output_has_nonzero_size(self):
        path = export_mp3(_sine(duration=1.0), SR_24, self.tmp / "out.mp3")
        self.assertGreater(path.stat().st_size, 500)

    def test_creates_parent_dirs(self):
        deep = self.tmp / "x" / "y" / "out.mp3"
        path = export_mp3(_sine(), SR_24, deep)
        self.assertTrue(path.exists())

    def test_different_bitrates_produce_different_sizes(self):
        tone = _sine(duration=3.0)
        lo = export_mp3(tone, SR_24, self.tmp / "lo.mp3", bitrate="64k")
        hi = export_mp3(tone, SR_24, self.tmp / "hi.mp3", bitrate="320k")
        self.assertLess(lo.stat().st_size, hi.stat().st_size)


# ---------------------------------------------------------------------------
# merge_audio_chunks
# ---------------------------------------------------------------------------

class TestMergeAudioChunks(unittest.TestCase):

    def test_empty_list_returns_empty_array(self):
        out = merge_audio_chunks([])
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(len(out), 0)

    def test_single_chunk_no_silence_added(self):
        tone = _sine(duration=1.0)
        out = merge_audio_chunks([tone], silence_ms=300, sample_rate=SR_24)
        np.testing.assert_array_equal(out, tone)

    def test_two_chunks_correct_total_length(self):
        A = _sine(duration=1.0, sr=SR_24)
        B = _sine(duration=1.0, sr=SR_24, freq=880.0)
        silence_samples = int(SR_24 * 0.3)
        out = merge_audio_chunks([A, B], silence_ms=300, sample_rate=SR_24)
        expected = len(A) + silence_samples + len(B)
        self.assertEqual(len(out), expected)

    def test_silence_between_chunks_is_zero(self):
        A = _sine(duration=0.5)
        B = _sine(duration=0.5, freq=880.0)
        out = merge_audio_chunks([A, B], silence_ms=200, sample_rate=SR_24)
        # silence zone is in the middle
        start = len(A)
        end   = start + int(SR_24 * 0.2)
        silence_zone = out[start:end]
        self.assertTrue(np.all(silence_zone == 0.0))

    def test_no_silence_after_last_chunk(self):
        A = np.ones(SR_24, dtype=np.float32)
        B = np.ones(SR_24, dtype=np.float32) * 2
        out = merge_audio_chunks([A, B], silence_ms=100, sample_rate=SR_24)
        # last sample must be from B (non-zero)
        self.assertNotEqual(out[-1], 0.0)

    def test_output_is_float32(self):
        out = merge_audio_chunks([_sine(), _sine(freq=880.0)], sample_rate=SR_24)
        self.assertEqual(out.dtype, np.float32)

    def test_three_chunks_correct_length(self):
        chunks = [_sine(duration=0.5) for _ in range(3)]
        silence_samples = int(SR_24 * 0.1)
        out = merge_audio_chunks(chunks, silence_ms=100, sample_rate=SR_24)
        expected = sum(len(c) for c in chunks) + 2 * silence_samples
        self.assertEqual(len(out), expected)

    def test_mixed_sample_rates_resampled_to_target(self):
        # EN chunk at 24 kHz, JA chunk at 44.1 kHz → merge at 24 kHz
        en_chunk = _sine(sr=SR_24, duration=1.0)
        ja_chunk = _sine(sr=SR_44, duration=1.0, freq=880.0)
        out = merge_audio_chunks(
            [en_chunk, ja_chunk],
            silence_ms=200,
            sample_rate=SR_24,
            chunk_sample_rates=[SR_24, SR_44],
        )
        # JA chunk resampled: 44100 samples → 24000 samples (≈ same duration)
        silence_samples = int(SR_24 * 0.2)
        expected = len(en_chunk) + silence_samples + SR_24   # JA is 1 s at 24k
        self.assertAlmostEqual(len(out), expected, delta=2)

    def test_mixed_rates_output_duration_correct(self):
        en = _sine(sr=SR_24, duration=2.0)
        ja = _sine(sr=SR_44, duration=1.0, freq=880.0)
        out = merge_audio_chunks(
            [en, ja], silence_ms=500, sample_rate=SR_24,
            chunk_sample_rates=[SR_24, SR_44],
        )
        total_dur = len(out) / SR_24
        # 2s + 0.5s silence + 1s JA = 3.5s
        self.assertAlmostEqual(total_dur, 3.5, places=1)


# ---------------------------------------------------------------------------
# get_duration
# ---------------------------------------------------------------------------

class TestGetDuration(unittest.TestCase):

    def test_one_second_sine(self):
        tone = _sine(sr=SR_24, duration=1.0)
        self.assertAlmostEqual(get_duration(tone, SR_24), 1.0, places=4)

    def test_fractional_duration(self):
        tone = _sine(sr=SR_44, duration=2.5)
        self.assertAlmostEqual(get_duration(tone, SR_44), 2.5, places=3)

    def test_empty_array_is_zero(self):
        self.assertEqual(get_duration(np.array([], dtype=np.float32), SR_24), 0.0)

    def test_raises_on_invalid_sample_rate(self):
        with self.assertRaises(ValueError):
            get_duration(_sine(), sample_rate=0)

    def test_returns_float(self):
        self.assertIsInstance(get_duration(_sine(), SR_24), float)


# ---------------------------------------------------------------------------
# validate_output_dir
# ---------------------------------------------------------------------------

class TestValidateOutputDir(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_creates_directory_if_not_exists(self):
        new_dir = self.tmp / "new_dir"
        self.assertFalse(new_dir.exists())
        result = validate_output_dir(new_dir)
        self.assertTrue(result.is_dir())

    def test_creates_nested_directories(self):
        deep = self.tmp / "a" / "b" / "c"
        result = validate_output_dir(deep)
        self.assertTrue(result.is_dir())

    def test_returns_path_object(self):
        result = validate_output_dir(self.tmp)
        self.assertIsInstance(result, Path)

    def test_existing_dir_does_not_raise(self):
        # Already exists — should succeed silently
        result = validate_output_dir(self.tmp)
        self.assertTrue(result.is_dir())

    def test_raises_if_path_is_a_file(self):
        file_path = self.tmp / "file.txt"
        file_path.touch()
        with self.assertRaises(NotADirectoryError):
            validate_output_dir(file_path)

    def test_accepts_string_path(self):
        result = validate_output_dir(str(self.tmp))
        self.assertIsInstance(result, Path)

    def test_returned_path_is_writable(self):
        result = validate_output_dir(self.tmp / "writable")
        probe = result / "probe.txt"
        probe.write_text("ok")
        self.assertTrue(probe.exists())
        probe.unlink()


if __name__ == "__main__":
    unittest.main(verbosity=2)
