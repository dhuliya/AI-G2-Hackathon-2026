from query_helper import AudioRAGQuerier
from eval_deterministic import run_deterministic_evals
from eval_llm import run_llm_evals

def print_summary(results):
    total_passed = 0
    for name, status, score, reason in results:
        status_flag = "[PASS]" if status == "PASSED" else "[FAIL]"
        print(f"{status_flag} {name}")
        print(f"       Score  : {score:.2f}")
        print(f"       Reason : {reason}\n")
        if status == "PASSED":
            total_passed += 1
    return total_passed

if __name__ == "__main__":
    print("==================================================")
    print("      STARTING AUDIO RAG EVALUATION SUITE        ")
    print("==================================================")

    # Initialize RAG and Model
    rag = AudioRAGQuerier(rag_store="./rag_store")

    # 1. Run Deterministic Suite
    print("\n>>> Phase 1: Running Deterministic Evaluations")
    deterministic_results = run_deterministic_evals(rag)

    # Combined Summary
    all_results = deterministic_results

    print("\n==================================================")
    print("               EVALUATION SUMMARY                 ")
    print("==================================================")
    
    passed_count = print_summary(all_results)

    print(f"Overall Result: {passed_count}/{len(all_results)} Scenarios Passed.")
    print("==================================================")