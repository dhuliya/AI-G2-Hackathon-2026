from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric

class ChronologicalOrderMetric(BaseMetric):
    """Scenario 2: Verifies returned segments are in monotonically increasing order."""
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.success = False
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase):
        segments = test_case.additional_metadata.get("segments", [])
        if not segments:
            self.score = 0.0
            self.reason = "No segments retrieved."
            self.success = False
            return self.score

        out_of_order_count = 0
        for i in range(len(segments) - 1):
            if segments[i]["start_sec"] > segments[i + 1]["start_sec"]:
                out_of_order_count += 1

        total_pairs = max(1, len(segments) - 1)
        self.score = (total_pairs - out_of_order_count) / total_pairs
        self.success = self.score >= self.threshold
        self.reason = f"{out_of_order_count} out-of-order transitions detected out of {total_pairs}."
        return self.score

    def is_successful(self) -> bool:
        return self.success


class TimeRangeBoundaryMetric(BaseMetric):
    """Scenario 3: Verifies all returned segments fall within [min_sec, max_sec]."""
    def __init__(self, min_sec: float, max_sec: float, threshold: float = 1.0):
        self.min_sec = min_sec
        self.max_sec = max_sec
        self.threshold = threshold
        self.success = False
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase):
        segments = test_case.additional_metadata.get("segments", [])
        if not segments:
            self.score = 1.0
            self.success = True
            self.reason = "Zero segments returned for range (valid)."
            return self.score

        out_of_bounds = 0
        for seg in segments:
            if seg["start_sec"] > self.max_sec or seg["end_sec"] < self.min_sec:
                out_of_bounds += 1

        self.score = (len(segments) - out_of_bounds) / len(segments)
        self.success = self.score >= self.threshold
        self.reason = f"{out_of_bounds} segments fell outside range [{self.min_sec}s - {self.max_sec}s]."
        return self.score

    def is_successful(self) -> bool:
        return self.success


class SpeakerIdentityMetric(BaseMetric):
    """Scenario 4: Verifies 100% of returned segments belong to target speaker."""
    def __init__(self, target_speaker, threshold: float = 1.0):
        self.target_speaker = str(target_speaker)
        self.threshold = threshold
        self.success = False
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase):
        segments = test_case.additional_metadata.get("segments", [])
        if not segments:
            self.score = 0.0
            self.reason = "No speaker segments retrieved."
            self.success = False
            return self.score

        mismatches = sum(1 for seg in segments if str(seg.get("speaker")) != self.target_speaker)
        self.score = (len(segments) - mismatches) / len(segments)
        self.success = self.score >= self.threshold
        self.reason = f"{mismatches} segments mismatched expected Speaker {self.target_speaker}."
        return self.score

    def is_successful(self) -> bool:
        return self.success