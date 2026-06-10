from rag.retriever import _rrf


def test_rrf_prefers_doc_ranked_high_in_both_lists():
    rankings = [[2, 0, 1], [0, 1, 2]]
    scores = _rrf(rankings, [1.0, 1.0], 60)
    assert max(scores, key=scores.get) == 0


def test_rrf_weights_shift_the_winner():
    rankings = [[0, 1], [1, 0]]
    scores = _rrf(rankings, [2.0, 1.0], 60)
    assert scores[0] > scores[1]
