import faiss
import numpy as np

# 示例文档向量（假设已经通过某种方式转换为向量表示）
document_vectors = np.array(
    [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4], [0.3, 0.4, 0.1]], dtype="float32"
)
print(document_vectors.shape)

# 查询向量
query_vector = np.array([[0.1, 0.2, 0.3]], dtype="float32")

# 创建FAISS索引
dimension = document_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)  # 使用L2距离（欧几里得距离）

# 添加向量到索引
index.add(document_vectors)

# 查询最相似的向量
k = 2  # 返回最相似的2个向量
distances, indices = index.search(query_vector, k)

# 输出结果
print("Query Vector:", query_vector)
print("Indices of most similar vectors:", indices)
print("Distances to most similar vectors:", distances)
