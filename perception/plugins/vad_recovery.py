"""Conservative physical endpoint check for very quiet, pending speech."""

from collections import deque
import math


class QuietEndpointer:
    FRAME_SAMPLES = 512
    LOW_RMS = 10 ** (-44 / 20)

    def __init__(self, sample_rate, silence_samples):
        self._pending = []
        self._recent = deque(maxlen=max(1, math.ceil(.25 * sample_rate / self.FRAME_SAMPLES)))
        self._silence_samples = silence_samples
        self._active = False
        self._peak_rms = 0.0
        self._quiet_samples = 0
        self._observation_peak = 0.0

    def observe(self, samples):
        self._observation_peak = 0.0
        self._pending.extend(samples)
        while len(self._pending) >= self.FRAME_SAMPLES:
            frame = self._pending[:self.FRAME_SAMPLES]
            del self._pending[:self.FRAME_SAMPLES]
            rms = math.sqrt(sum(value * value for value in frame) / self.FRAME_SAMPLES)
            self._recent.append(rms)
            self._observation_peak = max(self._observation_peak, rms)
            if self._active:
                self._peak_rms = max(self._peak_rms, rms)
                # A quieter syllable is still speech. Only near-digital silence
                # may override a model that continues to hold its segment open.
                if rms <= 1.0 / 32768:
                    self._quiet_samples += self.FRAME_SAMPLES
                else:
                    self._quiet_samples = 0

    def begin(self):
        self._active = True
        self._peak_rms = max(max(self._recent, default=0.0), self._observation_peak)
        self._quiet_samples = 0

    def end(self):
        self._active = False
        self._peak_rms = 0.0
        self._quiet_samples = 0

    @property
    def quiet(self):
        return self._active and self._peak_rms < self.LOW_RMS

    def ready(self):
        # Give the model two of its configured silence windows to finish.
        # No speech is invented from a silent input or steady low-level noise.
        return (self.quiet and self._peak_rms > 1.0 / 32768 and
                self._quiet_samples >= max(2 * self._silence_samples, self.FRAME_SAMPLES))
