try:
    from .paths import project_path, relative_path, model_reference
except ImportError:
    from paths import project_path, relative_path, model_reference

import asyncio
from typing import Any, List, Optional, Union
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from PIL import ImageOps

from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.base.embeddings.base import Embedding

from colpali_engine.models import ColQwen2, ColQwen2Processor, ColPali, ColPaliProcessor


def weighted_mean_pooling(hidden, attention_mask):
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    reps = s / d
    return reps

class VL_Embedding(MultiModalEmbedding):

    model: str = Field(description="The Multi-model to use.")

    api_key: Optional[str] = Field(
        default=None,
        description="The API key.",
    )
    dimensions: Optional[int] = Field(
        default=1024,
        description=(
            "The number of dimensions the resulting output embeddings should have. "
            "Only supported in embedding-3 and later models. embedding-2 is fixed at 1024."
        ),
    )
    timeout: Optional[float] = Field(
        default=None,
        description="The timeout.",
    )

    mode: str = Field(
        default='text',
        description="The mode of the model, either 'text' or 'image'."
    )

    # Hard cap on the longer side of input images to avoid OOM on very large pages.
    # This is particularly important for ColQwen2/ColPali style models where the
    # number of visual tokens (and memory) grows quickly with image resolution.
    max_image_side: int = Field(
        default=1536,
        description="Max length (pixels) of the longer side of an image before resizing for embedding.",
    )
    show_progress: bool = Field(
        default=False,
        description="Whether to show progress bars.",
    )
    
    embed_model: Union[ColQwen2, AutoModel, None] = Field(
        default=None
    )
    processor: Optional[ColQwen2Processor] = Field(
        default=None
    )
    tokenizer: Optional[AutoTokenizer] = Field(
        default=None
    )
    
    
    def __init__(
        self,
        model: str = "vidore/colqwen2-v1.0",
        dimensions: Optional[int] = 1024,
        timeout: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        mode: str = 'text',
        max_image_side: int = 1536,
        **kwargs: Any,
    ) -> None:
        model = model_reference(model)
        super().__init__(
            model=model,
            dimensions=dimensions,
            timeout=timeout,
            callback_manager=callback_manager,
            **kwargs,
        )
        
        self.mode = mode
        self.max_image_side = int(max_image_side) if max_image_side else 1536
        print("setting to image mode!!")
        if 'openbmb' in model:
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            self.embed_model = AutoModel.from_pretrained(model,
             torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map='cuda:0').cuda().eval()
            # self.embed_model.eval()
        elif ('vidore' in model and 'qwen' in model) or ('colqwen2' in model.lower() or 'colqwen' in model.lower()):
            # 支持本地路径，如果路径中包含 colqwen2 或 colqwen，也按照 ColQwen2 处理
            self.embed_model = ColQwen2.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                device_map='cuda:0',  # or "mps" if on Apple Silicon
            ).eval()
            self.processor = ColQwen2Processor.from_pretrained(model)
        elif 'vidore' in model and 'pali' in model:
            self.embed_model = ColPali.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                device_map='cuda:0',  # or "mps" if on Apple Silicon
            ).eval()
            self.processor = ColPaliProcessor.from_pretrained(model)
        else:
            # 如果都不匹配，尝试默认作为 ColQwen2 处理（适用于本地路径）
            print(f"Warning: Model name '{model}' doesn't match known patterns, trying ColQwen2...")
            try:
                self.embed_model = ColQwen2.from_pretrained(
                    model,
                    torch_dtype=torch.bfloat16,
                    device_map='cuda:0',
                ).eval()
                self.processor = ColQwen2Processor.from_pretrained(model)
            except Exception as e:
                raise ValueError(f"Failed to load model from '{model}': {e}. Please check the model path or use a supported model name.")
        
        print(f"Model loaded on device: {self.embed_model.device}")

    @staticmethod
    def _resize_to_max_side(img: Image.Image, max_side: int) -> Image.Image:
        """Resize image to ensure max(width, height) <= max_side while preserving aspect ratio."""
        if not max_side or max_side <= 0:
            return img
        w, h = img.size
        m = max(w, h)
        if m <= max_side:
            return img
        scale = max_side / float(m)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        
        


    @classmethod
    def class_name(cls) -> str:
        return "VL_Embedding"
    
    def embed_img(self, img_path):
        if isinstance(img_path, str):
            img_path = [img_path]
        img_path = [str(project_path(p)) for p in img_path]
        if 'vidore' in self.model or 'colqwen' in self.model.lower() or 'colpali' in self.model.lower():
            # Load + normalize + resize to reduce the risk of huge intermediate allocations.
            images = []
            for p in img_path:
                im = Image.open(p)
                im = ImageOps.exif_transpose(im)
                if im.mode != "RGB":
                    im = im.convert("RGB")
                im = self._resize_to_max_side(im, self.max_image_side)
                images.append(im)

            def _run_with_images(imgs: List[Image.Image]):
                batch_images = self.processor.process_images(imgs).to(self.embed_model.device)
                try:
                    with torch.no_grad():
                        return self.embed_model(**batch_images)
                finally:
                    # 清理中间变量以释放内存
                    del batch_images
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            try:
                image_embeddings = _run_with_images(images)
            except torch.cuda.OutOfMemoryError as e:
                # Retry once with a smaller cap.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                retry_side = max(512, int(self.max_image_side * 0.66))
                print(
                    f"CUDA OOM in embed_img, retrying with smaller max_image_side={retry_side}. "
                    f"Original error: {str(e)[:200]}",
                    flush=True,
                )
                images_retry = [self._resize_to_max_side(im, retry_side) for im in images]
                image_embeddings = _run_with_images(images_retry)
        elif 'openbmb' in self.model:
            images = [Image.open(img).convert('RGB') for img in img_path]
            inputs = {
                "text": [''] * len(images),
                'image': images,
                'tokenizer': self.tokenizer
            }
            with torch.no_grad():
                outputs = self.embed_model(**inputs)
                attention_mask = outputs.attention_mask
                hidden = outputs.last_hidden_state
                reps = weighted_mean_pooling(hidden, attention_mask)   
                image_embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
                # image_embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().tolist()[0]
            # image_embeddings = embeddings.tolist()[0]
        else:
             # Default fallback if logic allows, or raise error. 
             # Assuming similar to colqwen if we reached here from __init__
             images = [Image.open(img) for img in img_path]
             batch_images = self.processor.process_images(images).to(self.embed_model.device)
             with torch.no_grad():
                image_embeddings = self.embed_model(**batch_images)
        return image_embeddings
    
    def embed_text(self, text):
        if isinstance(text, str):
            text = [text]
        if 'colqwen' in self.model.lower() or 'vidore' in self.model:
            batch_queries = self.processor.process_queries(text).to(self.embed_model.device)
            with torch.no_grad():
                query_embeddings = self.embed_model(**batch_queries)
        elif 'colpali' in self.model.lower():
            batch_queries = self.processor.process_queries(text).to(self.embed_model.device)
            with torch.no_grad():
                query_embeddings = self.embed_model(**batch_queries)
        elif 'openbmb' in self.model:
            INSTRUCTION = "Represent this query for retrieving relevant documents: "
            queries = [INSTRUCTION + query for query in text]
            inputs = {
                "text": queries,
                'image': [None] * len(queries),
                'tokenizer': self.tokenizer
                }
            with torch.no_grad():
                outputs = self.embed_model(**inputs)
                attention_mask = outputs.attention_mask
                hidden = outputs.last_hidden_state
                reps = weighted_mean_pooling(hidden, attention_mask)   
                # query_embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
                query_embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().tolist()
                # query_embeddings = embeddings.tolist()[0]
        else:
            # Fallback
            batch_queries = self.processor.process_queries(text).to(self.embed_model.device)
            with torch.no_grad():
                query_embeddings = self.embed_model(**batch_queries)
        return query_embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.embed_text(query)[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.embed_text(text)[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embeddings_list: List[List[float]] = []
        for text in texts:
            embeddings = self.embed_text(text)
            embeddings = embeddings[0]
            embeddings_list.append(embeddings)
        return embeddings_list

    def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.embed_text(query)[0]
    
    def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.embed_text(text)[0]
    
    def _get_image_embedding(self, img_file_path) -> Embedding:
        return self.embed_img(img_file_path)
    
    def _aget_image_embedding(self, img_file_path) -> Embedding:
        return self.embed_img(img_file_path)
    
    def __call__(self, nodes, **kwargs):
        print(f"VL_Embedding __call__: Processing {len(nodes)} nodes")
        # for i, node in enumerate(nodes):
        #    print(f"Node {i} metadata: {node.metadata}")
        
        img_path = [str(project_path(p)) for p in img_path]
        if 'vidore' in self.model or 'colqwen' in self.model.lower() or 'colpali' in self.model.lower() or True: # Fallback to true if others fail to match but we want to try
            if self.mode == 'image':
                file_paths = [node.metadata['file_path'] for node in nodes]
                print(f"File paths: {file_paths[:5]}...") # Limit printing
                embeddings = self.embed_img(file_paths)
                # Ensure embeddings is a tensor before view
                if not isinstance(embeddings, torch.Tensor):
                     embeddings = torch.tensor(embeddings)
                embeddings = embeddings.view(embeddings.size(0),-1).tolist()
            else:
                embeddings = self.embed_text([node.text for node in nodes])
                if not isinstance(embeddings, torch.Tensor):
                     embeddings = torch.tensor(embeddings)
                embeddings = embeddings.view(embeddings.size(0),-1).tolist()

            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding
                
        elif 'openbmb' in self.model:
            if self.mode == 'image':
                file_paths = [node.metadata['file_path'] for node in nodes]
                print(f"File paths: {file_paths}")
                embeddings = self.embed_img(file_paths)
                embeddings = embeddings.tolist()
            else:
                embeddings = self.embed_text([node.text for node in nodes])
                # embeddings = embeddings.tolist()
                # embeddings = [embeddings]

            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding
                
        return nodes
    
    def score(self,image_embeddings,text_embeddings):
        img_path = [str(project_path(p)) for p in img_path]
        if 'vidore' in self.model or 'colqwen' in self.model.lower() or 'colpali' in self.model.lower():
            score = self.processor.score_multi_vector(image_embeddings, text_embeddings)
        elif 'openbmb' in self.model:
            score = text_embeddings @ image_embeddings.T
        else:
             score = self.processor.score_multi_vector(image_embeddings, text_embeddings)
        return score

if __name__ == "__main__":
    colpali = VL_Embedding("vidore/colqwen2-v1.0")
    image_embeddings = colpali.embed_img("corpus/slideVQA/images/3f5be9482ef37f7c7cb9be3e92d9d7a7_3.jpg")
    text_embeddings = colpali.embed_text("Hello, world!")
    score = colpali.processor.score_multi_vector(image_embeddings, text_embeddings)
    print(score)