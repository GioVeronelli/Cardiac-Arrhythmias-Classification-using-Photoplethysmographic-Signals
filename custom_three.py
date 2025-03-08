from sklearn.metrics import precision_score, recall_score

def custom_metric_threeclass(y_true, y_pred, precision_threshold=0.9, penalties=None):
    """
    Custom metric for three-class classification. Balances precision for the Normal class
    with recall for the Supraventricular and Ventricular classes.
    """
    # Calculate precision and recall for each class
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    
    # Set default penalties if not provided
    if penalties is None:
        penalties = [precision[0], recall[1], recall[2]]
    
    # Initialize weighted score
    score = 0
    
    # Penalize based on thresholds
    if precision[0] < precision_threshold:
        score += recall[1] * penalties[0] # Supraventricular recall penalized
        score += recall[2] * penalties[0] # Ventricular recall penalized
    else:
        score += recall[1]  # Supraventricular recall unpenalized
        score += recall[2]  # Ventricular recall unpenalized
    
    # Average score for the two abnormal classes
    return round(score / 2, 2)