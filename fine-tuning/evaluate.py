import math
import re

_span_pat = re.compile(
    r'(?i)interacted\s+with\s+items(.*?)in\s+chronological\s+order\.',
    re.S,
)

def get_topk_results(predictions, scores, targets, k, all_items=None):
    results = []
    B = len(targets)
    predictions = [_.split("Response:")[-1] for _ in predictions]
    predictions = [_.strip().replace(" ","") for _ in predictions]

    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]

        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        print(f'sorted pairs:{sorted_pairs}')
        target_item = targets[b]
        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] == target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)

    return results


def extract_items_string(text):
    m = _span_pat.search(text)
    if m:
        return m.group(1).strip()    
    

def get_detailed_topk_results(predictions, scores, targets, k, all_items=None):
    detailed_results = {}
    results = []
    pred_list =[]
    input_list = []

    B = len(targets)
    for b in range(B):
        first_each_beam = extract_items_string(predictions[b*k])
        input_list.append(first_each_beam)

    predictions = [_.split("Response:")[-1] for _ in predictions]
    predictions = [_.strip().replace(" ","") for _ in predictions]

    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]

        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        target_item = targets[b]
        one_results = []
        pred_results = []
        for sorted_pred in sorted_pairs:
            pred_results.append(sorted_pred[0])

            if sorted_pred[0] == target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)
        pred_list.append(pred_results)
    
    detailed_results['results'] = results
    detailed_results['input_list'] = input_list
    detailed_results['pred_list'] = pred_list

    return detailed_results


def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError

    return res


def ndcg_k(topk_results, k):

    ndcg = 0.0
    for row in topk_results:
        res = row[:k]
        one_ndcg = 0.0
        for i in range(len(res)):
            one_ndcg += res[i] / math.log(i + 2, 2)
        ndcg += one_ndcg
    return ndcg


def hit_k(topk_results, k):
    hit = 0.0
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            hit += 1
    return hit

