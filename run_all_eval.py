from query_helper import AudioRAGQuerier
from gemini_wrapper import GoogleGemini
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
    gemini_model = GoogleGemini(model_name="gemini-3.5-flash")

    # 1. Run Deterministic Suite
    print("\n>>> Phase 1: Running Deterministic Evaluations")
    deterministic_results = run_deterministic_evals(rag)

    # 2. Run LLM Suite
    print("\n>>> Phase 2: Running LLM-Graded Evaluations")
    llm_results = run_llm_evals(rag, gemini_model)

    print("\n>>> Phase 3: Running Hybrid Evaluations")
    from eval_hybrid import run_test_scenario
    run_test_scenario()

    # Combined Summary
    all_results = deterministic_results + llm_results

    print("\n==================================================")
    print("               EVALUATION SUMMARY                 ")
    print("==================================================")
    
    passed_count = print_summary(all_results)

    print(f"Overall Result: {passed_count}/{len(all_results)} Scenarios Passed.")
    print("==================================================")