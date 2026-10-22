try:
    from .paths import project_path, relative_path, model_reference, record_paths
except ImportError:
    from paths import project_path, relative_path, model_reference, record_paths

# Parse CLI options before importing GPU libraries, so --help stays lightweight.
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Build FinRAGBench visual retrieval indexes")
    parser.add_argument('--input_dirs', nargs='+', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--embed_model_name', required=True)
    parser.add_argument('--total_files', type=int, default=6000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.total_files < 1 or args.workers < 1:
        parser.error('--total_files and --workers must be positive')

import os
import sys
import json
import random
import torch
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.core import SimpleDirectoryReader

if __package__:
    from .vl_embedding import VL_Embedding
else:
    from vl_embedding import VL_Embedding

class Ingestion:
    def __init__(
        self,
        dataset_dir=None,
        input_prefix='img',
        output_dir='corpus/finragbench/',
        embed_model_name='vidore/colqwen2-v1.0'
    ):
        self.dataset_dir = str(project_path(dataset_dir)) if dataset_dir is not None else None
        dataset_dir = self.dataset_dir
        self.input_dir = os.path.join(dataset_dir, input_prefix) if dataset_dir is not None else None
        self.output_dir = str(project_path(output_dir))
        self.workers = 4  # 默认4个worker，可以根据GPU内存调整
        self.embed_model_name = embed_model_name
        self.max_image_size = 16384  # 最大图片尺寸警告阈值，超过此尺寸会警告但不跳过（让程序尝试处理，OOM会被捕获）
        # 不在初始化时创建 reader，而是在 ingestion_example 中按需创建
        self.pipeline = IngestionPipeline(transformations=[
            SimpleFileNodeParser(),
            VL_Embedding(model=embed_model_name, mode='image')
        ])
        # 添加GPU锁，避免多线程竞争
        self.gpu_lock = threading.Lock()



    def ingestion_example(self, input_file, output_file):
        # 检查图片大小，如果太大则警告但不跳过（让程序尝试处理，如果OOM会被捕获）
        try:
            with Image.open(input_file) as img:
                width, height = img.size
                max_dimension = max(width, height)
                if max_dimension > self.max_image_size:
                    print(f"⚠ Large image detected: {os.path.basename(input_file)} ({width}x{height}, max_dim={max_dimension}), will attempt to process (may cause OOM)")
                # 不再跳过，让程序尝试处理
        except Exception as e:
            # 如果无法检查图片大小，继续处理（可能是非图片文件）
            pass

        # 使用GPU锁确保同一时间只有一个线程使用GPU
        with self.gpu_lock:
            try:
                # 为每个文件创建新的 reader，传入文件所在的目录
                file_dir = os.path.dirname(input_file)
                reader = SimpleDirectoryReader(input_dir=file_dir, input_files=[input_file])
                documents = reader.load_data()

                if not documents:
                    print(f"Warning: No documents loaded from {input_file}")
                    return False

                print(f"Processing: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
                try:
                    nodes = self.pipeline.run(documents=documents, num_workers=1, show_progress=False)
                except torch.cuda.OutOfMemoryError as e:
                    print(f"CUDA OOM error processing {os.path.basename(input_file)}: {str(e)}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return False
                except Exception as e:
                    print(f"Error in pipeline.run for {os.path.basename(input_file)}: {str(e)}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return False

                # 处理完成后立即清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if not nodes:
                    print(f"Warning: No nodes generated from {os.path.basename(input_file)}")
                    return False

                print(f"Generated {len(nodes)} nodes from {os.path.basename(input_file)}")
                nodes_json = [node.to_dict() for node in nodes]

                # 确保输出目录存在
                output_dir = os.path.dirname(output_file)
                os.makedirs(output_dir, exist_ok=True)

                print(f"Writing to: {output_file}")
                with open(output_file, 'w') as json_file:
                    json.dump(record_paths(nodes_json, relative=True), json_file, indent=2, ensure_ascii=False)

                # 验证文件是否成功创建
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    if file_size > 0:
                        print(f"✓ Successfully created {os.path.basename(output_file)} ({file_size} bytes)")
                        return True
                    else:
                        print(f"✗ Output file is empty: {output_file}")
                        return False
                else:
                    print(f"✗ Output file not created: {output_file}")
                    return False
            except Exception as e:
                print(f"✗ Error processing {os.path.basename(input_file)}: {str(e)}")
                import traceback
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False

    def ingestion_multi_session(self):
        os.makedirs(self.output_dir, exist_ok=True)
        file_to_process = []

        # 递归遍历子文件夹
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                # 只处理图片文件
                if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
                    continue

                input_file = os.path.join(root, file)
                file_prefix, _ = os.path.splitext(file)

                # 保持相对路径结构，避免文件名冲突
                rel_path = os.path.relpath(input_file, self.input_dir)
                rel_dir = os.path.dirname(rel_path)
                rel_file_prefix = os.path.splitext(os.path.basename(rel_path))[0]

                # 在输出目录中保持相同的目录结构
                if rel_dir:
                    output_subdir = os.path.join(self.output_dir, rel_dir)
                    os.makedirs(output_subdir, exist_ok=True)
                    output_file = os.path.join(output_subdir, rel_file_prefix) + '.node'
                else:
                    output_file = os.path.join(self.output_dir, rel_file_prefix) + '.node'

                if not os.path.exists(output_file):
                    file_to_process.append((input_file, output_file))

        if self.workers == 1:
            for input_file, output_file in tqdm(file_to_process):
                self.ingestion_example(input_file, output_file)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_file = {executor.submit(self.ingestion_example, input_file, output_file): (input_file, output_file) for input_file, output_file in file_to_process}
                for future in tqdm(as_completed(future_to_file), total=len(file_to_process), desc='Processing files'):
                    result_type = future.result()

    def ingestion_from_multiple_dirs(self, input_dirs, total_files=6000, seed=42):
        """
        从多个目录中选择文件进行索引构建

        Args:
            input_dirs: 输入目录列表，例如 ['data/finragbench/ch', 'data/finragbench/en']
            total_files: 总共要处理的文件数量
            seed: 随机种子，确保可重复性
        """
        os.makedirs(self.output_dir, exist_ok=True)
        random.seed(seed)

        # 收集所有候选文件
        all_candidates = []
        for input_dir in map(lambda p: str(project_path(p)), input_dirs):
            print(f"Scanning directory: {input_dir}")
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    # 只处理图片文件
                    if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
                        continue

                    input_file = os.path.join(root, file)
                    # 确定输出文件路径，使用目录名作为前缀避免冲突
                    dir_name = os.path.basename(input_dir)  # ch 或 en
                    rel_path = os.path.relpath(input_file, input_dir)
                    rel_file_prefix = os.path.splitext(os.path.basename(rel_path))[0]

                    # 在输出目录中创建子目录（ch或en）
                    output_subdir = os.path.join(self.output_dir, dir_name)
                    os.makedirs(output_subdir, exist_ok=True)
                    output_file = os.path.join(output_subdir, rel_file_prefix) + '.node'

                    # 只添加未处理的文件
                    if not os.path.exists(output_file):
                        all_candidates.append((input_file, output_file, dir_name))

        print(f"Found {len(all_candidates)} candidate files")

        # 随机选择指定数量的文件
        if len(all_candidates) > total_files:
            selected_files = random.sample(all_candidates, total_files)
            print(f"Randomly selected {total_files} files from {len(all_candidates)} candidates")
        else:
            selected_files = all_candidates
            print(f"Using all {len(selected_files)} candidate files (less than {total_files})")

        # 统计各目录的文件数量
        dir_counts = {}
        for _, _, dir_name in selected_files:
            dir_counts[dir_name] = dir_counts.get(dir_name, 0) + 1
        print(f"File distribution: {dir_counts}")

        # 处理文件
        file_to_process = [(in_file, out_file) for in_file, out_file, _ in selected_files]
        print(f"\nStarting to process {len(file_to_process)} files with {self.workers} workers...")
        print(f"Output directory: {self.output_dir}")

        if self.workers == 1:
            success_count = 0
            for idx, (input_file, output_file) in enumerate(tqdm(file_to_process, desc='Processing files'), 1):
                if self.ingestion_example(input_file, output_file):
                    success_count += 1
                if idx % 10 == 0:
                    print(f"Progress: {idx}/{len(file_to_process)} files processed, {success_count} successful")
            print(f"\nCompleted: {success_count}/{len(file_to_process)} files processed successfully")
        else:
            success_count = 0
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_file = {executor.submit(self.ingestion_example, input_file, output_file): (input_file, output_file) for input_file, output_file in file_to_process}
                for idx, future in enumerate(tqdm(as_completed(future_to_file), total=len(file_to_process), desc='Processing files'), 1):
                    result = future.result()
                    if result:
                        success_count += 1
                    if idx % 50 == 0:
                        print(f"Progress: {idx}/{len(file_to_process)} files processed, {success_count} successful, {idx-success_count} failed")
            print(f"\nCompleted: {success_count}/{len(file_to_process)} files processed successfully")


if __name__ == '__main__':
    ingestion = Ingestion(
        dataset_dir=None,
        output_dir=args.output_dir,
        embed_model_name=model_reference(args.embed_model_name),
    )
    ingestion.workers = args.workers
    ingestion.ingestion_from_multiple_dirs(
        args.input_dirs, total_files=args.total_files, seed=args.seed)
