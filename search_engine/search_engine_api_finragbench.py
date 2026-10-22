try:
    from .paths import project_path, relative_path, model_reference
except ImportError:
    from paths import project_path, relative_path, model_reference

from fastapi import FastAPI, Query
import uvicorn
from typing import List, Dict, Any
from collections import defaultdict
from pathlib import Path
import argparse
import os
import json

# FinRAGBench 数据路径
finragbench_json_path = str(project_path("data/finragbench/citation_labels_new/selected_200_with_bboxes.json"))
image_dir = str(project_path("data/finragbench/citation_labels_new/sampled_imgs/"))
ingestion_dir = str(project_path("corpus/finragbench/"))
embed_model_name = "models/colqwen2-v0.1-merged"

# FinRAGBench sample storage
finragbench_examples: Dict[str, Dict[str, Any]] = {}
finragbench_examples_by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

def load_finragbench_metadata():
    """加载 FinRAGBench json 数据"""
    global finragbench_examples, finragbench_examples_by_image
    print(f"Loading FinRAGBench annotations from {finragbench_json_path}...")
    if not os.path.exists(finragbench_json_path):
        print(f"FinRAGBench JSON not found: {finragbench_json_path}")
        return

    try:
        with open(finragbench_json_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except Exception as e:
        print(f"Failed to read FinRAGBench JSON: {e}")
        return

    # FinRAGBench JSON 格式：直接是数组
    examples = payload if isinstance(payload, list) else []
    if not isinstance(examples, list):
        print(f"Warning: Expected list format, got {type(payload)}")
        examples = []

    print(f"Found {len(examples)} examples in JSON file")

    for example in examples:
        # FinRAGBench 使用 query-id 作为唯一标识
        uid = example.get('query-id')
        if not uid:
            continue
        finragbench_examples[uid] = example

        # 从 img_paths 中提取图片路径信息
        img_paths = example.get('img_paths', {})
        from_pages = example.get('from_pages', '')

        # img_paths 是一个字典，key 是页码，value 是相对路径
        # 例如: {"125": "./sampled_imgs/11463b6562c6ba2afde67a29bca314452ea9a2e0_125.png-3/11463b6562c6ba2afde67a29bca314452ea9a2e0_125.png"}
        for page, img_path in img_paths.items():
            # 提取文件名（去掉路径和扩展名）
            img_file = os.path.basename(img_path)
            img_name_without_ext = os.path.splitext(img_file)[0]

            # 构建多个可能的 key 用于匹配
            # 1. 完整的文件名（不含扩展名）
            finragbench_examples_by_image[img_name_without_ext].append(example)

            # 2. 如果文件名包含下划线，也尝试去掉最后一部分（如去掉 "-3"）
            if '_' in img_name_without_ext:
                parts = img_name_without_ext.split('_')
                if len(parts) > 1:
                    # 尝试去掉最后一部分（可能是页码或序号）
                    base_key = '_'.join(parts[:-1])
                    finragbench_examples_by_image[base_key].append(example)

            # 3. 使用 page 作为 key
            if page:
                finragbench_examples_by_image[str(page)].append(example)

            # 4. 使用 file_name_page 格式（类似 vidoseek）
            if from_pages:
                # 尝试从文件名提取基础名称
                if '_' in img_name_without_ext:
                    base_parts = img_name_without_ext.rsplit('_', 1)
                    if len(base_parts) == 2:
                        base_name = base_parts[0]
                        key = f"{base_name}_{page}"
                        finragbench_examples_by_image[key].append(example)

    print(f"Loaded {len(finragbench_examples)} examples, {len(finragbench_examples_by_image)} image keys.")

def get_image_path(image_name: str) -> str:
    """寻找对应的图像路径（支持递归搜索子目录）

    搜索顺序：
    1. 原始 image_dir (citation_labels_new/sampled_imgs/)
    2. data/finragbench/en/
    3. data/finragbench/ch/
    """
    if not image_name:
        return None

    if os.path.isabs(image_name) and os.path.exists(image_name):
        return image_name

    # 搜索目录列表（按优先级）
    search_dirs = [
        image_dir,  # 原始目录
        "data/finragbench/en/",
        "data/finragbench/ch/",
    ]

    base_name = os.path.splitext(image_name)[0]

    # 在每个搜索目录中查找
    for search_dir in map(lambda p: str(project_path(p)), search_dirs):
        if not os.path.exists(search_dir):
            continue

        # 首先尝试直接路径
        candidate = os.path.join(search_dir, image_name)
        if os.path.exists(candidate):
            return candidate

        # 如果直接路径不存在，递归搜索子目录
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                file_base = os.path.splitext(file)[0]
                if file_base == base_name or file == image_name:
                    found_path = os.path.join(root, file)
                    if os.path.exists(found_path):
                        return found_path

    return None

# 创建 FastAPI 应用实例
app = FastAPI(
    title="FinRAGBench Search Engine API",
    description="Provides search functionality for the FinRAGBench benchmark.",
    version="1.0.0",
)

# 全局变量，用于存储 SearchEngine 实例
search_engine = None

@app.on_event("startup")
async def startup_event():
    global search_engine
    print("Initializing FinRAGBench SearchEngine...")
    load_finragbench_metadata()
    # 使用 ingestion_dir 作为 dataset_dir，node_dir_prefix 设为空字符串
    # 因为 ingestion_dir 已经是完整的 node 目录路径
    # FinRAGBench ingestion writes nested en/ch index directories.
    if __package__:
        from .search_engine import SearchEngine, nodefile2node
    else:
        from search_engine.search_engine import SearchEngine, nodefile2node

    class FinRAGSearchEngine(SearchEngine):
        def load_nodes(self):
            files = sorted(Path(self.node_dir).rglob('*.node'))
            if not files:
                raise RuntimeError(f"No .node files in {self.node_dir}")
            return [node for path in files for node in nodefile2node(path)]

    search_engine = FinRAGSearchEngine(
        dataset_dir=ingestion_dir,
        node_dir_prefix='',  # 空字符串，因为 dataset_dir 已经是完整的 node 目录
        embed_model_name=embed_model_name
    )


@app.get("/health")
def health():
    return {"status": "ok" if search_engine is not None else "starting",
            "metadata_examples": len(finragbench_examples)}


# 定义搜索 API 端点
@app.get(
    "/search",
    summary="Perform a search query.",
    description="Executes a search using the initialized SearchEngine and returns the results.",
    response_model=List[List[Dict[str, Any]]]  # 定义响应模型，提高 API 文档的清晰度
)
async def search(queries: List[str] = Query(...)):
    """
    执行搜索操作。

    Args:
        query: 搜索查询字符串。

    Returns:
        搜索结果列表。
    """
    if search_engine is None:
        return [[] for _ in queries]

    try:
        print(f"Received queries: {queries}")
        results_batch = search_engine.batch_search(queries)
        print(f"Search results: {results_batch}")

        processed_results = []
        for query_results in results_batch:
            query_processed = []
            for idx, file_name in enumerate(query_results):
                image_path = get_image_path(file_name)
                # 尝试多种方式匹配 image_key
                image_key = os.path.splitext(file_name)[0]
                # 也尝试去掉可能的子目录路径
                image_key_base = os.path.basename(image_key)

                matches = finragbench_examples_by_image.get(image_key, [])
                if not matches:
                    matches = finragbench_examples_by_image.get(image_key_base, [])

                primary_example = matches[0] if matches else None

                # FinRAGBench 数据格式适配
                reference_page = []
                if primary_example:
                    from_pages = primary_example.get('from_pages', '')
                    if from_pages:
                        # from_pages 可能是字符串，如 "125" 或 "125,143"
                        if isinstance(from_pages, str):
                            pages = [p.strip() for p in from_pages.split(',') if p.strip()]
                            reference_page = pages
                        elif isinstance(from_pages, list):
                            reference_page = [str(p) for p in from_pages]
                        else:
                            reference_page = [str(from_pages)]

                # 获取 bounding_box
                bounding_box = None
                if primary_example:
                    bboxes = primary_example.get('bboxes', {})
                    if bboxes and reference_page:
                        # bboxes 是一个字典，key 是页码
                        page_bboxes = bboxes.get(str(reference_page[0])) if reference_page else None
                        if page_bboxes and len(page_bboxes) > 0:
                            # 取第一个 bbox，格式是 {"xmin": 0.118, "ymin": 0.147, "xmax": 0.893, "ymax": 0.19}
                            bbox = page_bboxes[0]
                            # 转换为 [x1, y1, x2, y2] 格式（需要知道图片尺寸才能转换，这里先保存原始格式）
                            bounding_box = bbox

                detail = {
                    'idx': idx,
                    'image_name': file_name,
                    'image_id': Path(file_name).stem,
                    'image_file': relative_path(image_path) if image_path else f"missing_{file_name}",
                    'match_count': len(matches),
                    'uid': primary_example.get('query-id') if primary_example else None,
                    'query': primary_example.get('query') if primary_example else None,
                    'reference_answer': primary_example.get('answer') if primary_example else None,
                    'reference_page': reference_page,
                    'from_pages': primary_example.get('from_pages', '') if primary_example else '',
                    'bounding_box': bounding_box,
                    'bboxes': primary_example.get('bboxes', {}) if primary_example else {},
                    'category': primary_example.get('category') if primary_example else None,
                    'answer_type': primary_example.get('answer_type') if primary_example else None,
                }
                query_processed.append(detail)

            processed_results.append(query_processed)

        print(f"Processed results: {len(processed_results)} queries processed")
        return processed_results
    except Exception as e:
        print(f"Error in search endpoint: {e}")
        import traceback
        traceback.print_exc()
        return [[]]

def main():
    global finragbench_json_path, image_dir, ingestion_dir, embed_model_name
    parser = argparse.ArgumentParser(description="FinRAGBench retrieval API")
    parser.add_argument('--metadata', default=finragbench_json_path)
    parser.add_argument('--image_root', default=image_dir)
    parser.add_argument('--index_dir', default=ingestion_dir)
    parser.add_argument('--embed_model_name', default=embed_model_name)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8005)
    args = parser.parse_args()
    finragbench_json_path = str(project_path(args.metadata))
    image_dir = str(project_path(args.image_root))
    ingestion_dir = str(project_path(args.index_dir))
    embed_model_name = model_reference(args.embed_model_name)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
