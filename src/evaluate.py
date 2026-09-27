import re


def normalize_answer(text):
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = " ".join(text.split())
    return text


def exact_match(prediction, ground_truth):
    return int(
        normalize_answer(prediction)
        == normalize_answer(ground_truth)
    )


def token_f1(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if not prediction_tokens or not ground_truth_tokens:
        return int(prediction_tokens == ground_truth_tokens)

    common = set(prediction_tokens) & set(ground_truth_tokens)

    if not common:
        return 0.0

    common_count = sum(
        min(
            prediction_tokens.count(token),
            ground_truth_tokens.count(token)
        )
        for token in common
    )

    precision = common_count / len(prediction_tokens)
    recall = common_count / len(ground_truth_tokens)

    return (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )


def retrieval_recall(retrieved_documents, gold_context):
    """
    Measures whether the actual gold SQuAD context was retrieved.

    Returns:
        1 if the gold context is present among retrieved documents,
        otherwise 0.
    """

    normalized_gold = normalize_answer(gold_context)

    for document in retrieved_documents:

        normalized_document = normalize_answer(document)

        # Exact context match
        if normalized_document == normalized_gold:
            return 1

    return 0