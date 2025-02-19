import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))


import argparse
import json
import os
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.signal import savgol_filter


from utils.dataset_loader import *


def evaluate_inference(inference_results):
    total = len(inference_results)
    correct_num = sum(r['is_correct'] == 1 for r in inference_results)
    accuracy = correct_num / total if total > 0 else 0.0
    return total, accuracy, correct_num


def smooth_data(data, window_length=5, polyorder=2):
    import numpy as np
    from scipy.signal import savgol_filter

    data_arr = np.array(data, dtype=float)

    if len(data_arr) < window_length:
        return data_arr
    return savgol_filter(data_arr, window_length=window_length, polyorder=polyorder)


def group_and_aggregate_responses(
    results_confidence,
    input_file
):
    grouped_results = defaultdict(list)
    for result in results_confidence:
        if result['answer']:
            grouped_results[result['prompt']].append(result)

    methods_count = 9
    pbar = tqdm(total=methods_count, desc="Aggregation Methods")

    pbar.set_description("Step 1/9: Threshold-based Early Stopping")
    thresholds = np.arange(0.0, 1.01, 0.01)
    threshold_based_accuracies = []
    threshold_based_avg_used = []

    for thr in thresholds:
        final_results_thr = []
        total_checks = 0

        for prompt, responses in grouped_results.items():
            found = False
            for i, r in enumerate(responses):
                if r['confidence'] >= thr:
                    final_results_thr.append(r)
                    total_checks += (i + 1)
                    found = True
                    break
            if not found:
                weighted_scores = defaultdict(float)
                for r in responses:
                    weighted_scores[r['answer']] += r['confidence']
                final_answer_weighted = max(weighted_scores, key=weighted_scores.get)
                final_r = next(r for r in responses if r['answer'] == final_answer_weighted)
                final_results_thr.append(final_r)
                total_checks += len(responses)

        _, acc_thr, _ = evaluate_inference(final_results_thr)
        threshold_based_accuracies.append(acc_thr)
        threshold_based_avg_used.append(total_checks / len(grouped_results))

    pbar.update(1)

    pbar.set_description("Step 2/9: ES (Weighted Dynamic Voting)")
    es_accuracies = []
    es_avg_used = []

    for thr in thresholds:
        final_results_es = []
        total_checks_es = 0

        for prompt, responses in grouped_results.items():
            used = 0
            done = False
            weighted_scores = defaultdict(float)

            for i, r in enumerate(responses):
                used += 1
                weighted_scores[r['answer']] += r['confidence']

                if i >= 1:
                    sum_conf = sum(weighted_scores.values())
                    top_answer = max(weighted_scores, key=weighted_scores.get)
                    top_conf = weighted_scores[top_answer]
                    ratio = top_conf / sum_conf if sum_conf > 0 else 0.0

                    if ratio >= thr:
                        final_choice = next(
                            x for x in responses[: i + 1] if x['answer'] == top_answer
                        )
                        final_results_es.append(final_choice)
                        total_checks_es += used
                        done = True
                        break

            if not done:
                top_answer = max(weighted_scores, key=weighted_scores.get)
                final_choice = next(r for r in responses if r['answer'] == top_answer)
                final_results_es.append(final_choice)
                total_checks_es += used

        _, acc_es, _ = evaluate_inference(final_results_es)
        es_accuracies.append(acc_es)
        es_avg_used.append(total_checks_es / len(grouped_results))

    pbar.update(1)

    pbar.set_description("Step 3/9: Adaptive Self-Consistency")
    asc_accuracies = []
    asc_avg_used = []

    for thr in thresholds:
        final_results_asc = []
        total_checks_asc = 0

        for prompt, responses in grouped_results.items():
            used = 0
            done = False
            vote_counts = defaultdict(int)

            for i, r in enumerate(responses):
                used += 1
                vote_counts[r['answer']] += 1

                if i >= 1:
                    total_count = sum(vote_counts.values())
                    top_answer = max(vote_counts, key=vote_counts.get)
                    top_count = vote_counts[top_answer]
                    ratio = top_count / total_count if total_count > 0 else 0.0

                    if ratio >= thr:
                        final_choice = next(
                            x for x in responses[: i + 1] if x['answer'] == top_answer
                        )
                        final_results_asc.append(final_choice)
                        total_checks_asc += used
                        done = True
                        break

            if not done:
                freq_counts = defaultdict(int)
                for r in responses:
                    freq_counts[r['answer']] += 1
                best_ans = max(freq_counts, key=freq_counts.get)
                final_r = next(r for r in responses if r['answer'] == best_ans)
                final_results_asc.append(final_r)
                total_checks_asc += used

        _, acc_asc, _ = evaluate_inference(final_results_asc)
        asc_accuracies.append(acc_asc)
        asc_avg_used.append(total_checks_asc / len(grouped_results))

    pbar.update(1)

    pbar.set_description("Step 4/9: Self-Consistency (n=1..max)")
    max_responses_per_prompt = max(len(v) for v in grouped_results.values()) if grouped_results else 0
    self_consistency_n_accuracies = []
    self_consistency_n_avg_used = []

    sc_n_candidates = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1024]
    for n in sc_n_candidates:
        if n > max_responses_per_prompt:
            break
        final_results_sc_n = []
        total_used_sc_n = 0

        for prompt, responses in grouped_results.items():
            use_count = min(n, len(responses))
            sub_responses = responses[:use_count]

            freq_counts = defaultdict(int)
            for r in sub_responses:
                freq_counts[r['answer']] += 1
            most_frequent = max(freq_counts, key=freq_counts.get)
            final_r = next(r for r in sub_responses if r['answer'] == most_frequent)

            final_results_sc_n.append(final_r)
            total_used_sc_n += use_count

        _, acc_sc_n, _ = evaluate_inference(final_results_sc_n)
        self_consistency_n_accuracies.append(acc_sc_n)
        self_consistency_n_avg_used.append(total_used_sc_n / len(grouped_results))

    pbar.update(1)

    pbar.set_description("Step 5/9: Weighted Self-Consistency (n=1..max)")
    weighted_sc_n_accuracies = []
    weighted_sc_n_avg_used = []

    for n in sc_n_candidates:
        if n > max_responses_per_prompt:
            break
        final_results_wsc_n = []
        total_used_wsc_n = 0

        for prompt, responses in grouped_results.items():
            use_count = min(n, len(responses))
            sub_responses = responses[:use_count]

            w_scores = defaultdict(float)
            for r in sub_responses:
                w_scores[r['answer']] += r['confidence']
            best_ans = max(w_scores, key=w_scores.get)
            final_r = next(r for r in sub_responses if r['answer'] == best_ans)

            final_results_wsc_n.append(final_r)
            total_used_wsc_n += use_count

        _, acc_wsc_n, _ = evaluate_inference(final_results_wsc_n)
        weighted_sc_n_accuracies.append(acc_wsc_n)
        weighted_sc_n_avg_used.append(total_used_wsc_n / len(grouped_results))

    pbar.update(1)

    pbar.set_description("Step 6/9: Highest Confidence (n=1..max)")
    highest_confidence_n_accuracies = []
    highest_confidence_n_avg_used = []

    for n in sc_n_candidates:
        if n > max_responses_per_prompt:
            break
        final_results_hc_n = []
        total_used_hc_n = 0

        for prompt, responses in grouped_results.items():
            use_count = min(n, len(responses))
            sub_responses = responses[:use_count]

            best_response = max(sub_responses, key=lambda x: x['confidence'])
            final_results_hc_n.append(best_response)
            total_used_hc_n += use_count

        _, acc_hc_n, _ = evaluate_inference(final_results_hc_n)
        highest_confidence_n_accuracies.append(acc_hc_n)
        highest_confidence_n_avg_used.append(total_used_hc_n / len(grouped_results))

    pbar.update(1)

    pbar.set_description("Step 8/9: Sliding Window + SC")
    sliding_window_sizes = range(2, max_responses_per_prompt + 1)
    sliding_window_accuracies = []
    sliding_window_avg_used = []

    stop_sliding = False

    final_results_self_consistency = []
    total_used_self_consistency = 0

    for prompt, responses in grouped_results.items():
        freq_counts = defaultdict(int)
        for resp in responses:
            freq_counts[resp['answer']] += 1
        most_frequent = max(freq_counts, key=freq_counts.get)
        final_r = next(resp for resp in responses if resp['answer'] == most_frequent)
        final_results_self_consistency.append(final_r)
        total_used_self_consistency += len(responses)

    for w_size in sliding_window_sizes:
        if stop_sliding:
            break

        final_results_sw = []
        total_used_sw = 0

        for prompt, responses in grouped_results.items():
            used = 0
            window = []
            final_r = None

            for r in responses:
                used += 1
                window.append(r['answer'])
                if len(window) > w_size:
                    window.pop(0)

                if len(window) == w_size and len(set(window)) == 1:
                    final_r = r
                    break

            if final_r is None:
                stop_sliding = True
                break

            final_results_sw.append(final_r)
            total_used_sw += used

        if stop_sliding:
            final_results_sw = final_results_self_consistency
            total_used_sw = total_used_self_consistency

        _, acc_sw, _ = evaluate_inference(final_results_sw)
        avg_used_sw = total_used_sw / len(grouped_results)

        sliding_window_accuracies.append(acc_sw)
        sliding_window_avg_used.append(avg_used_sw)

    pbar.update(1)

    pbar.set_description("Step 9/9: pass@N")
    pass_n_accuracies = []
    pass_n_avg_used = []

    for n in sc_n_candidates:
        if n > max_responses_per_prompt:
            break
        pass_count = 0
        total_used = 0

        for prompt, responses in grouped_results.items():
            use_count = min(n, len(responses))
            sub_responses = responses[:use_count]

            if any(r['is_correct'] == 1 for r in sub_responses):
                pass_count += 1

            total_used += use_count

        pass_rate = pass_count / len(grouped_results) if grouped_results else 0.0
        pass_n_accuracies.append(pass_rate)
        pass_n_avg_used.append(total_used / len(grouped_results))

    pbar.update(1)
    pbar.close()

    threshold_based_accuracies_pct = [v * 100 for v in threshold_based_accuracies]
    es_accuracies_pct = [v * 100 for v in es_accuracies]
    asc_accuracies_pct = [v * 100 for v in asc_accuracies]
    self_consistency_n_accuracies_pct = [v * 100 for v in self_consistency_n_accuracies]
    weighted_sc_n_accuracies_pct = [v * 100 for v in weighted_sc_n_accuracies]
    highest_confidence_n_accuracies_pct = [v * 100 for v in highest_confidence_n_accuracies]
    sliding_window_accuracies_pct = [v * 100 for v in sliding_window_accuracies]
    pass_n_accuracies_pct = [v * 100 for v in pass_n_accuracies]

    threshold_based_accuracies_pct_smooth  = smooth_data(threshold_based_accuracies_pct, 5, 2)
    es_accuracies_pct_smooth               = smooth_data(es_accuracies_pct, 5, 2)
    asc_accuracies_pct_smooth              = smooth_data(asc_accuracies_pct, 5, 2)
    self_consistency_n_accuracies_pct_smooth = smooth_data(self_consistency_n_accuracies_pct, 5, 2)
    weighted_sc_n_accuracies_pct_smooth    = smooth_data(weighted_sc_n_accuracies_pct, 5, 2)
    highest_confidence_n_accuracies_pct_smooth = smooth_data(highest_confidence_n_accuracies_pct, 5, 2)
    sliding_window_accuracies_pct_smooth   = smooth_data(sliding_window_accuracies_pct, 5, 2)

    plt.figure(figsize=(8, 5))

    plt.plot(
        threshold_based_avg_used, threshold_based_accuracies_pct_smooth,
        color='tab:blue', label="Early stopping"
    )

    plt.plot(
        es_avg_used, es_accuracies_pct_smooth,
        color='tab:orange', label="ASC w/ conf."
    )

    plt.plot(
        asc_avg_used, asc_accuracies_pct_smooth,
        color='tab:gray', label="ASC"
    )

    plt.plot(
        self_consistency_n_avg_used, self_consistency_n_accuracies_pct_smooth,
        color='tab:green', label="SC"
    )

    plt.plot(
        weighted_sc_n_avg_used, weighted_sc_n_accuracies_pct_smooth,
        color='tab:red', label="SC w/ conf."
    )

    plt.plot(
        highest_confidence_n_avg_used, highest_confidence_n_accuracies_pct_smooth,
        color='tab:purple', label="Best-of-n"
    )

    plt.plot(
        sliding_window_avg_used, sliding_window_accuracies_pct_smooth,
        color='tab:brown', label="ESC"
    )


    all_acc_values_pct = (
        threshold_based_accuracies_pct_smooth.tolist()
        + es_accuracies_pct_smooth.tolist()
        + asc_accuracies_pct_smooth.tolist()
        + self_consistency_n_accuracies_pct_smooth.tolist()
        + weighted_sc_n_accuracies_pct_smooth.tolist()
        + highest_confidence_n_accuracies_pct_smooth.tolist()
        + sliding_window_accuracies_pct_smooth.tolist()
    )
    if len(all_acc_values_pct) > 0:
        acc_min, acc_max = min(all_acc_values_pct), max(all_acc_values_pct)
    else:
        acc_min, acc_max = 0, 100

    def round_down_5pct(x):
        return math.floor(x * 100 / 200) * 200 / 100

    def round_up_5pct(x):
        return math.ceil(x * 100 / 200) * 200 / 100

    y_min = round_down_5pct(acc_min)
    y_max = round_up_5pct(acc_max)
    y_min = max(y_min, 0)
    y_max = min(y_max, 100) if y_max <= 100 else y_max

    plt.ylim([y_min, y_max])
    plt.xscale("log", base=2)
    plt.xlabel("Sample Budgets", fontsize=18)
    plt.ylabel("Accuracy(%)", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=12)

    pdf_output_path = os.path.join(input_file, 'compare_experiment.pdf')
    plt.savefig(pdf_output_path, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"[INFO] Comparison figure saved to {pdf_output_path}")

    data_last = {
        "Method": [
            "Early stopping",
            "ASC w/ conf.",
            "ASC",
            "SC",
            "SC w/ conf.",
            "Best-of-n",
            "ESC",
            "pass@N"
        ],
        "Accuracy(%)": [
            threshold_based_accuracies_pct[-1] if threshold_based_accuracies_pct else 0,
            es_accuracies_pct[-1] if es_accuracies_pct else 0,
            asc_accuracies_pct[-1] if asc_accuracies_pct else 0,
            self_consistency_n_accuracies_pct[-1] if self_consistency_n_accuracies_pct else 0,
            weighted_sc_n_accuracies_pct[-1] if weighted_sc_n_accuracies_pct else 0,
            highest_confidence_n_accuracies_pct[-1] if highest_confidence_n_accuracies_pct else 0,
            sliding_window_accuracies_pct[-1] if sliding_window_accuracies_pct else 0,
            pass_n_accuracies_pct[-1] if pass_n_accuracies_pct else 0,
        ],
        "Avg_Responses_Used": [
            threshold_based_avg_used[-1] if threshold_based_avg_used else 0,
            es_avg_used[-1] if es_avg_used else 0,
            asc_avg_used[-1] if asc_avg_used else 0,
            self_consistency_n_avg_used[-1] if self_consistency_n_avg_used else 0,
            weighted_sc_n_avg_used[-1] if weighted_sc_n_avg_used else 0,
            highest_confidence_n_avg_used[-1] if highest_confidence_n_avg_used else 0,
            sliding_window_avg_used[-1] if sliding_window_avg_used else 0,
            pass_n_avg_used[-1] if pass_n_avg_used else 0,
        ]
    }

    df_last = pd.DataFrame(data_last)
    csv_output_path_last = os.path.join(input_file, 'aggregation_results.csv')
    df_last.to_csv(csv_output_path_last, index=False, encoding='utf-8')
    print(f"[INFO] Aggregation results saved to {csv_output_path_last}")


    pass_x = pass_n_avg_used
    pass_y = pass_n_accuracies
    pass_at_1 = pass_y[0] if pass_y else 0.0

    methods_for_score = {
        "Early stopping":     (threshold_based_avg_used,   threshold_based_accuracies),
        "ASC w/ conf.":       (es_avg_used,                es_accuracies),
        "ASC":                (asc_avg_used,               asc_accuracies),
        "SC":                 (self_consistency_n_avg_used, self_consistency_n_accuracies),
        "SC w/ conf.":        (weighted_sc_n_avg_used,     weighted_sc_n_accuracies),
        "Best-of-n":          (highest_confidence_n_avg_used, highest_confidence_n_accuracies),
        "ESC":                (sliding_window_avg_used,    sliding_window_accuracies),
    }

    sample_x_positions = [2, 4, 16, 64, 256, 1024]

    def find_value_at_x(target_x, x_array, y_array):

        idx = None
        for i, val in enumerate(x_array):
            if val <= target_x:
                idx = i
            else:
                break
        if idx is None:
            return None
        return y_array[idx]

    score_table_rows = []

    for x_val in sample_x_positions:
        row_data = {"X": x_val}
        passN = find_value_at_x(x_val, pass_x, pass_y)
        if passN is None:
            passN = None

        for method_name, (mx, my) in methods_for_score.items():
            perfN = find_value_at_x(x_val, mx, my)
            if passN is None or perfN is None:
                row_data[method_name] = None
            else:
                row_data[method_name] = round(100 * perfN, 2)

        score_table_rows.append(row_data)

    df_score = pd.DataFrame(score_table_rows)
    metric_csv_path = os.path.join(input_file, 'metric.csv')
    df_score.to_csv(metric_csv_path, index=False, encoding='utf-8')
    print(f"[INFO] Score_N table saved to {metric_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Group responses by prompt, aggregate them, and compare methods by plotting #used vs. accuracy (in %)."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the directory containing results_with_confidence.json, and where outputs will be saved."
    )
    parser.add_argument("--dataset_name", type=str, default=None)
    args = parser.parse_args()

    json_path_confidence = os.path.join(args.input_file, 'results_with_confidence.json')
    with open(json_path_confidence, 'r', encoding='utf-8') as f:
        data_conf = json.load(f)
    results_confidence = data_conf['results']

    if args.dataset_name:
        handler = get_dataset(args.dataset_name)
        d1 = []
        for result_confidence in results_confidence:
            result_confidence['answer'] = handler.extract_answer(result_confidence["response"])
            result_confidence['is_correct'] = handler.check(result_confidence['correct_answer'], result_confidence['answer'])
            d1.append(result_confidence)
        results_confidence = d1


    group_and_aggregate_responses(
        results_confidence,
        args.input_file
    )


if __name__ == "__main__":
    main()
