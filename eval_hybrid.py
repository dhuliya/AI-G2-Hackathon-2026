import shutil
import os
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi


def evaluate_hybrid_scenario(rag, llm_model):
    print("\n--- Running Hybrid Search Scenario ---")
    query = "oral history archive xylophone"
    matches = rag.hybrid_search(query=query, top_k=3)

    if not matches:
        return False, 0.0, "No search hits returned from hybrid search."

    retrieved_contexts = [m["transcript"] for m in matches]
    
    # Formatted actual output including rich metadata details
    formatted_output_lines = []
    for idx, m in enumerate(matches, start=1):
        line = (
            f"Hit #{idx} [RRF Score: {m['rrf_score']:.4f}] "
            f"[{m['start_sec']:.1f}s - {m['end_sec']:.1f}s] "
            f"Speaker {m['speaker_label']} (Blob: {m['audio_blob']}): {m['transcript']}"
        )
        formatted_output_lines.append(line)

    test_case = LLMTestCase(
        input=query,
        actual_output="\n".join(formatted_output_lines),
        retrieval_context=retrieved_contexts
    )
    
    metric = ContextualRelevancyMetric(threshold=0.7, model=llm_model)
    metric.measure(test_case)

    return metric.is_successful(), metric.score, metric.reason

def run_hybrid_evals(rag, llm_model):
    scenarios = [
        ("Scenario Hybrid: Hybrid RRF Search Relevancy", evaluate_hybrid_scenario),
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


## below is obsolated code.

# ==========================================
# 1. HYBRID SEARCH LOGIC TO EVALUATE
# ==========================================
def hybrid_search(query: str, collections: dict, top_k: int = 5, k: int = 60):
    trans_col = collections["transcribe_segments"]
    attr_col = collections["audio_attributes"]
    seg_col = collections["audio_segments"]

    # 1. Vector Search (Semantic)
    query_res = trans_col.query(query_texts=[query], n_results=top_k)
    vector_ids = query_res["ids"][0] if query_res["ids"] else []

    # 2. Keyword Search (BM25)
    all_data = trans_col.get()
    if not all_data["documents"]:
        return []

    bm25 = BM25Okapi([doc.lower().split() for doc in all_data["documents"]])
    bm25_scores = bm25.get_scores(query.lower().split())

    keyword_ids = [
        all_data["ids"][i]
        for i in sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    ]

    # 3. Simple RRF Scoring
    rrf_scores = {}

    for rank, doc_id in enumerate(vector_ids, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    for rank, doc_id in enumerate(keyword_ids, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # Sort top items by RRF score
    fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # 4. Fetch Details & Format Output
    results = []
    for trans_id, score in fused_results:
        # Get transcript text
        trans_data = trans_col.get(ids=[trans_id])
        if not trans_data["ids"]:
            continue

        transcript_text = trans_data["documents"][0]
        trans_meta = trans_data["metadatas"][0] if trans_data["metadatas"] and trans_data["metadatas"][0] else {}

        # DIRECT ID DERIVATION (Guarantees matching across collections)
        # "trans_audio_123_seg_0" -> "attr_audio_123_seg_0"
        # "trans_audio_123_seg_0" -> "audio_123_seg_0"
        attr_id = trans_id.replace("trans_", "attr_", 1)
        seg_id = trans_id.replace("trans_", "", 1)

        # Query audio_attributes safely
        attr_data = attr_col.get(ids=[attr_id])
        attr = (
            attr_data["metadatas"][0]
            if attr_data["metadatas"] and attr_data["metadatas"][0]
            else {}
        )

        # Query audio_segments safely
        seg_data = seg_col.get(ids=[seg_id])
        seg = (
            seg_data["metadatas"][0]
            if seg_data["metadatas"] and seg_data["metadatas"][0]
            else {}
        )

        # Extract with multi-level fallbacks (attr_col -> trans_meta -> default)
        start_sec = attr.get("start_sec") if "start_sec" in attr else trans_meta.get("start_sec")
        end_sec = attr.get("end_sec") if "end_sec" in attr else trans_meta.get("end_sec")
        speaker = attr.get("speaker_label", trans_meta.get("speaker_label", "Unknown"))
        blob = seg.get("blob_path", trans_meta.get("blob_path", "N/A"))

        results.append({
            "transcript": transcript_text,
            "rrf_score": score,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "speaker_label": speaker,
            "audio_blob": blob
        })

    return results


# ==========================================
# 2. TEST ENVIRONMENT SETUP
# ==========================================
def setup_test_environment():
    # Use ephemeral in-memory Chroma client for testing
    client = chromadb.EphemeralClient()
    
    # Simple default embedding function
    ef = embedding_functions.DefaultEmbeddingFunction()

    collections = {
        "transcribe_segments": client.create_collection("transcribe_segments", embedding_function=ef),
        "audio_attributes": client.create_collection("audio_attributes", embedding_function=ef),
        "audio_segments": client.create_collection("audio_segments", embedding_function=ef),
    }

    # Test Data setup:
    # Seg 0: Pure semantic match for "history and audio archives"
    # Seg 1: Pure exact keyword match for "xylophone pitch scale"
    # Seg 2: Dual match (Contains keywords 'xylophone' + semantic history context)
    test_segments = [
        {
            "seg_index": 0,
            "transcript": "We discussed old historical sound recordings and oral traditions in the archive.",
            "start": 0.0, "end": 10.5, "speaker": 1,
            "file": "/data/segments/test_seg_0.wav"
        },
        {
            "seg_index": 1,
            "transcript": "The xylophone pitch scale calibration frequency was set at 440 Hertz.",
            "start": 10.5, "end": 22.0, "speaker": 2,
            "file": "/data/segments/test_seg_1.wav"
        },
        {
            "seg_index": 2,
            "transcript": "In our historical oral history archive, we recorded a rare xylophone pitch acoustic session.",
            "start": 22.0, "end": 38.2, "speaker": 1,
            "file": "/data/segments/test_seg_2.wav"
        }
    ]

    audio_id = "audio_test_123"

    for seg in test_segments:
        segment_id = f"{audio_id}_seg_{seg['seg_index']}"
        trans_id = f"trans_{segment_id}"
        attr_id = f"attr_{segment_id}"

        # 1. Transcribe Collection
        collections["transcribe_segments"].add(
            ids=[trans_id],
            documents=[seg["transcript"]],
            metadatas=[{"audio_id": audio_id, "segment_id": segment_id}]
        )

        # 2. Attributes Collection
        collections["audio_attributes"].add(
            ids=[attr_id],
            documents=[f"Attributes for {segment_id}"],
            metadatas=[{
                "audio_id": audio_id,
                "segment_id": segment_id,
                "start_sec": seg["start"],
                "end_sec": seg["end"],
                "speaker_label": seg["speaker"]
            }]
        )

        # 3. Segments Collection
        collections["audio_segments"].add(
            ids=[segment_id],
            documents=[f"Audio slice {segment_id}"],
            metadatas=[{
                "audio_id": audio_id,
                "segment_id": segment_id,
                "blob_path": seg["file"]
            }]
        )

    return collections

from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric

def evaluate_hybrid_search_scenario(
    query: str, 
    collections: dict, 
    llm_model, 
    top_k: int = 3, 
    threshold: float = 0.7
):
    print(f"\n--- Running Hybrid Search Evaluation: '{query}' ---")
    
    # 1. Execute hybrid search
    matches = hybrid_search(query=query, collections=collections, top_k=top_k)
    
    if not matches:
        print("No matches returned from hybrid search.")
        return False, 0.0, "No search hits returned."

    # 2. Extract transcript contexts for DeepEval
    retrieved_contexts = [m["transcript"] for m in matches]
    
    # Format actual_output containing formatted transcript lines
    actual_output = "\n".join([
        f"Speaker {m.get('speaker_label', 'Unknown')}: {m['transcript']}"
        for m in matches
    ])

    # 3. Construct DeepEval LLMTestCase
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieved_contexts
    )

    # 4. Measure contextual relevancy using the LLM model
    metric = ContextualRelevancyMetric(threshold=threshold, model=llm_model)
    metric.measure(test_case)

    # 5. Print results & detailed hit metadata
    print(f"Evaluation Passed : {metric.is_successful()}")
    print(f"Relevancy Score   : {metric.score:.4f}")
    print(f"Reason            : {metric.reason}\n")
    
    print("--- Retrieved Hits Details ---")
    for idx, m in enumerate(matches, start=1):
        start = f"{m['start_sec']:.1f}s" if isinstance(m.get('start_sec'), (int, float)) else "N/A"
        end = f"{m['end_sec']:.1f}s" if isinstance(m.get('end_sec'), (int, float)) else "N/A"
        print(f"#{idx} [RRF Score: {m['rrf_score']:.4f}] [{start} - {end}] Speaker {m['speaker_label']}")
        print(f"    Transcript: {m['transcript']}")
        print(f"    Audio Path: {m['audio_blob']}\n")

    return metric.is_successful(), metric.score, metric.reason


# ==========================================
# 3. RUN TEST SUITE
# ==========================================
def run_test_scenario():
    print("Setting up mock database...")
    collections = setup_test_environment()

    # Query target: combines keyword ('xylophone') with concept ('oral history archive')
    query = "oral history archive xylophone"
    print(f"\n--- Executing Test Query: '{query}' ---")
    
    matches = hybrid_search(query, collections, top_k=3)

    print(f"\nRetrieved {len(matches)} results sorted by RRF Score:\n")

    # Assertions & Verification
    for idx, m in enumerate(matches, start=1):
        # Validate metadata fields are present and not None
        assert m["start_sec"] is not None, f"Failed: start_sec is None in {m['trans_id']}"
        assert m["end_sec"] is not None, f"Failed: end_sec is None in {m['trans_id']}"
        assert m["speaker_label"] != "Unknown", f"Failed: speaker_label missing in {m['trans_id']}"
        assert m["audio_blob"] != "N/A", f"Failed: audio_blob missing in {m['trans_id']}"

        print(f"Rank #{idx} | RRF Score: {m['rrf_score']:.5f}")
        print(f" ID: {m['trans_id']}")
        print(f" Timestamp: [{m['start_sec']:.1f}s - {m['end_sec']:.1f}s]")
        print(f" Speaker: {m['speaker_label']}")
        print(f" Audio Path: {m['audio_blob']}")
        print(f" Transcript: {m['transcript']}\n")

    print("ALL TEST ASSERTIONS PASSED SUCCESSFULLY!")

# Run test directly
if __name__ == "__main__" or "__file__" not in globals():
    run_test_scenario()