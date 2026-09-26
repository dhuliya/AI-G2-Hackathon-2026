from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric

def evaluate_scenario_1(rag, llm_model):
    print("\n--- Running Scenario 1: Semantic Search Relevance ---")
    query = "growing up in New York"
    search_hits = rag.semantic_search(query_text=query, n_results=3)
    retrieved_contexts = [hit["transcript"] for hit in search_hits]

    test_case = LLMTestCase(
        input=query,
        actual_output="\n".join(retrieved_contexts),
        retrieval_context=retrieved_contexts
    )
    metric = ContextualRelevancyMetric(threshold=0.7, model=llm_model)
    metric.measure(test_case)

    return metric.is_successful(), metric.score, metric.reason


def evaluate_scenario_5(rag, llm_model):
    print("\n--- Running Scenario 5: QA Context Window Retrieval ---")
    query = "where did the speaker grow up?"
    qa_data = rag.get_qa_context_window(query_text=query, n_before=2, m_after=2)

    retrieved_contexts = [seg["text"] for seg in qa_data["context_segments"]]
    formatted_context = qa_data["formatted_context"]

    test_case = LLMTestCase(
        input=query,
        actual_output=formatted_context,
        retrieval_context=retrieved_contexts
    )
    metric = ContextualRelevancyMetric(threshold=0.7, model=llm_model)
    metric.measure(test_case)

    return metric.is_successful(), metric.score, metric.reason


def run_llm_evals(rag, llm_model):
    scenarios = [
        ("Scenario 1: Semantic Search Relevance", evaluate_scenario_1),
        ("Scenario 5: QA Context Window Relevancy", evaluate_scenario_5),
    ]

    results = []
    for name, func in scenarios:
        try:
            passed, score, reason = func(rag, llm_model)
            status = "PASSED" if passed else "FAILED"
            results.append((name, status, score, reason))
        except Exception as e:
            results.append((name, "ERROR", 0.0, str(e)))

    return results
