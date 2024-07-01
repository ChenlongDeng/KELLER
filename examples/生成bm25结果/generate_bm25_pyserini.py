from pyserini.index import IndexCollection

# 定义索引参数
index_args = IndexCollection.Args()
index_args.input = 'path/to/jsonl'
index_args.collection = 'JsonCollection'
index_args.generator = 'DefaultLuceneDocumentGenerator'
index_args.index = 'path/to/index'
index_args.storePositions = True
index_args.storeDocvectors = True
index_args.storeRaw = True

# 执行索引
IndexCollection.index(index_args)

from pyserini.index import IndexReader

# 索引文件夹
index_dir = "path/to/index_directory"

# 创建索引
# 注意: 这需要使用 Pyserini 的 Java 工具来创建索引，具体操作请参阅 Pyserini 文档
from pyserini.search import SimpleSearcher

# 初始化搜索器
searcher = SimpleSearcher(index_dir)

# 执行查询
query = "快乐的狐狸"
hits = searcher.search(query)

# 打印结果
for i in range(0, len(hits)):
    print(f'{i+1:2} {hits[i].docid:10} {hits[i].score:.5f}')
