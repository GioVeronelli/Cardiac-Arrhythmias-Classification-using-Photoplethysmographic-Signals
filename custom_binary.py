from sklearn.metrics import precision_score, recall_score

def custom_metric_binary(y_true, y_pred, precision_threshold=0.9, penalty=None):
    """
    Custom metric for binary classification.
    """
    # Calculate precision for class 0 (Normal) and recall for class 1 (Abnormal)
    precision_class_0 = precision_score(y_true, y_pred, pos_label=0)
    recall_class_1 = recall_score(y_true, y_pred, pos_label=1)

    # If the precision for 'Normal' beats is below the threshold, apply a penalty to recall
    if precision_class_0 < precision_threshold:
        if penalty is None:
            penalty = precision_class_0
        return round(recall_class_1 * penalty, 2)

    # Otherwise, return the recall for 'Abnormal' beats
    return recall_class_1