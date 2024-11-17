import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 示例文档
documents = [
    "This is a sample document.",
    "This document is another sample.",
    "And this is a different document.",
]

# 查询
query = "sample document"

# 将文档和查询转换为TF-IDF向量
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)
query_vector = vectorizer.transform([query])

# 计算余弦相似度
similarity_scores = cosine_similarity(query_vector, doc_vectors).flatten()

# 根据相似度得分对文档进行排序
sorted_indices = np.argsort(similarity_scores)[::-1]
sorted_documents = [documents[i] for i in sorted_indices]

# 输出结果
print("Query:", query)
print("Documents ranked by similarity:")
for idx, doc in enumerate(sorted_documents):
    print(f"{idx + 1}: {doc} (Score: {similarity_scores[sorted_indices[idx]]:.4f})")
