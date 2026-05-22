import numpy as np

def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    predict = predict.astype(int)
    actual = actual.astype(int) 
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    beta = 1
    f1 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-6)
    return f1, precision, recall, TP, TN, FP, FN


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is larger than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = (score > threshold).astype(int)
    else:
        predict = pred.astype(int)

    adjusted_predict = predict.copy()
    latency = 0
    anomaly_count = 0

    events = get_event_indices(label)
    for start, end in events:
        if np.any(predict[start:end+1]):
            adjusted_predict[start:end+1] = 1
            latency += np.argmax(predict[start:end+1])
            anomaly_count += 1

    if calc_latency:
        return adjusted_predict, latency / (anomaly_count + 1e-4)
    else:
        return adjusted_predict
    

def get_event_indices(label):
    """ Returns the start-end index of a sequence of consecutive 1s (event unit). """
    events = []
    in_event = False
    start = 0
    for i, val in enumerate(label):
        if val and not in_event:
            in_event = True
            start = i
        elif not val and in_event:
            in_event = False
            events.append((start, i-1))
    if in_event:
        events.append((start, len(label)-1))
    return events


def calc_composite_f1(predict, label):
    """
    Composite F1 score calculation: Based on time-wise Precision + event-wise Recall

    Returns:
        Fc1, Precision_time, Recall_event, TP_time, FP_time, TP_event, FN_event
    """
    predict = predict.astype(int)
    label = label.astype(int)

    # Time-wise Precision
    TP_time = np.sum(predict * label)
    FP_time = np.sum(predict * (1-label))
    precision_time = TP_time / (TP_time + FP_time + 1e-6)
    
    predict = predict.astype(bool)
    label = label.astype(bool)
    
    # Event-wise Recall
    events = get_event_indices(label)
    TP_event = 0
    for start, end in events:
        if np.any(predict[start:end+1]):
            TP_event += 1
    FN_event = len(events) - TP_event
    recall_event = TP_event / (TP_event + FN_event + 1e-6)

    # Composite F1
    fc1 = 2 * precision_time * recall_event / (precision_time + recall_event + 1e-6)
    return fc1, precision_time, recall_event, TP_time, FP_time, TP_event, FN_event

    
def calc_seq(score, label, threshold, mode):
    score = np.asarray(score)
    label = np.asarray(label)

    if mode == 'point-wise':
        predict = score > threshold
        return calc_point2point(predict, label)
    elif mode == 'point-adjusted':
        predict = adjust_predicts(score, label, threshold, calc_latency=False)
        return calc_point2point(predict, label)
    elif mode == 'composite':
        predict = score > threshold
        return calc_composite_f1(predict, label)


def bf_search(score, label, start, end=None, step=1, verbose=True, mode='point-wise'):
    assert mode in ['point-wise', 'point-adjusted', 'composite'], f"Invalid mode: {mode}"
    assert len(score) == len(label), "score and label must have the same length!"

    if end is None:
        end = start
        step = 1
    
    if verbose:
        print(f"Search range: {start}-{end} (step={step})")
    
    threshold = start
    search_step = (end-start) / float(step)
    m = (-1., -1., -1.)
    m_t = 0.0
    metric_name = {
        'point-wise': 'F1',
        'point-adjusted': 'Adjusted F1',
        'composite': 'Composite F1'
    }[mode]

    for i in range(step):
        threshold += search_step
        target = calc_seq(score, label, threshold, mode)
        f1, precision, recall, TP, TN, FP, FN = target

        if f1 > m[0]:
            m = target
            m_t = threshold
        
            if verbose:
                print(f"[{i}/{step}] {metric_name}={m[0]:.4f}, Precision={m[1]:.4f}, Recall={m[2]:.4f}")

    if verbose:
        print(f"Result: {m}, {m_t}")
    return m, m_t

def calc_result(scores, labels, mode, verbose=True):
    # --- Best F1 threshold search ---
    if verbose:
        print(f"\nSearching best threshold for {mode} F1 score...")
        
    best_result, best_thr = bf_search(
        score=scores,
        label=labels,
        start=np.min(scores),
        end=np.max(scores),
        step=100,
        verbose=False,
        mode=mode
    )

    f1, precision, recall, tp, tn, fp, fn = best_result
    if verbose:
        print(f"\n=== Final Result ({mode}) ===")
        print(f"Best Threshold: {best_thr:.6f}")
        print(f"Precision:      {precision:.4f}")
        print(f"Recall:         {recall:.4f}")
        print(f"F1 Score:       {f1:.4f}")
        print(f"TP / TN / FP / FN:   {tp} / {tn} / {fp} / {fn}")

    return best_result, best_thr