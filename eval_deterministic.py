from deepeval.test_case import LLMTestCase
from deterministic_metrics import (
    ChronologicalOrderMetric,
    TimeRangeBoundaryMetric,
    SpeakerIdentityMetric,
)

def evaluate_scenario_2(rag):
    print("\n--- Running Scenario 2: Chronological Ordering ---")
    segments = rag.get_ordered_segments()

    test_case = LLMTestCase(
        input="Get full ordered transcript",
        actual_output=f"Retrieved {len(segments)} segments",
        additional_metadata={"segments": segments}
    )
    metric = ChronologicalOrderMetric(threshold=1.0)
    metric.measure(test_case)

    return metric.is_successful(), metric.score, metric.reason


def evaluate_scenario_3(rag):
    print("\n--- Running Scenario 3: Time Range Boundary Filtering ---")
    min_sec, max_sec = 0.0, 60.0
    segments = rag.get_segments_by_time_range(min_sec=min_sec, max_sec=max_sec)

    test_case = LLMTestCase(
        input=f"Filter time range [{min_sec}s - {max_sec}s]",
        actual_output=f"Retrieved {len(segments)} segments",
        additional_metadata={"segments": segments}
    )
    metric = TimeRangeBoundaryMetric(min_sec=min_sec, max_sec=max_sec, threshold=1.0)
    metric.measure(test_case)

    return metric.is_successful(), metric.score, metric.reason


def evaluate_scenario_4(rag):
    print("\n--- Running Scenario 4: Speaker Dialogue Filtering ---")
    target_speaker = 1
    segments = rag.get_speaker_dialogue(speaker_label=target_speaker)

    test_case = LLMTestCase(
        input=f"Filter for Speaker {target_speaker}",
        actual_output=f"Retrieved {len(segments)} segments",
        additional_metadata={"segments": segments}
    )
    metric = SpeakerIdentityMetric(target_speaker=target_speaker, threshold=1.0)
    metric.measure(test_case)

    return metric.is_successful(), metric.score, metric.reason


def run_deterministic_evals(rag):
    scenarios = [
        ("Scenario 2: Chronological Order", evaluate_scenario_2),
        ("Scenario 3: Time Range Filter", evaluate_scenario_3),
        ("Scenario 4: Speaker Dialogue Filter", evaluate_scenario_4),
    ]

    results = []
    for name, func in scenarios:
        try:
            passed, score, reason = func(rag)
            status = "PASSED" if passed else "FAILED"
            results.append((name, status, score, reason))
        except Exception as e:
            results.append((name, "ERROR", 0.0, str(e)))

    return results