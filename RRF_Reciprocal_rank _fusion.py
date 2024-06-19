from collections import defaultdict

def reciprocal_rank_fusion(ranked_lists, constant=60):
    """
    Thực hiện hợp nhất thứ hạng hồi quy (Reciprocal Rank Fusion - RRF) trên nhiều danh sách xếp hạng.

    :param ranked_lists: Danh sách các danh sách xếp hạng, mỗi danh sách xếp hạng là một danh sách các tài liệu ID.
    :param constant: Giá trị hằng số để kiểm soát ảnh hưởng của thứ hạng (mặc định là 60).
    :return: Danh sách các tài liệu ID được sắp xếp theo điểm RRF giảm dần.
    """
    # Khởi tạo từ điển để lưu điểm RRF
    rrf_scores = defaultdict(float)

    # Duyệt qua từng danh sách xếp hạng
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            # rank + 1 để thứ hạng bắt đầu từ 1 thay vì 0
            rrf_scores[doc_id] += 1 / (rank + 1 + constant)

    # Sắp xếp các tài liệu theo điểm RRF giảm dần
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    print(sorted_docs)
    # Trả về danh sách các tài liệu ID đã sắp xếp
    return [doc_id for doc_id, score in sorted_docs]

# Ví dụ sử dụng
ranked_lists = [
    ['doc1', 'doc2', 'doc3', 'doc4'],
    ['doc3', 'doc1', 'doc4', 'doc2'],
    ['doc2', 'doc4', 'doc1', 'doc3']
]

final_ranking = reciprocal_rank_fusion(ranked_lists)
print(final_ranking)
