#!/usr/bin/env python
"""
CUDA_VISIBLE_DEVICES=4,5,6 \
 torchrun --nproc_per_node=3 --master_port=29500 \
   embed_pro.py \
   --img-dir /data1/tzz/DIV4K/4KLSDB_zzh/train/HR \
   --webdataset-dir /data1/tzz/DIV4K/Sana/asset/dataset/training_data \
   --tile-size 1024 \
   --tile-overlap 128 \
   --scale 0.41407 \
   --batch-size 16 \
   --num-workers 1 \
   --precision bf16 \
   --text-batch-size 16 \
   --checkpoint-every 100 \
   --resume-from-checkpoint


   CUDA_VISIBLE_DEVICES=4 \
 torchrun --nproc_per_node=1 --master_port=29500 \
   embed_pro.py \
   --img-dir /data1/tzz/DIV4K/4KLSDB_zzh/train/HR \
   --webdataset-dir /data1/tzz/DIV4K/Sana/asset/dataset/training_data \
   --tile-size 1024 \
   --tile-overlap 128 \
   --scale 0.41407 \
   --batch-size 16 \
   --num-workers 1 \
   --precision bf16 \
   --text-batch-size 16 \
   --checkpoint-every 100 \
   --resume-from-checkpoint
"""

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import argparse
import io
import json
import math
import os
import pickle
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
import webdataset as wds
from PIL import Image
from diffusers import AutoencoderDC
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from diffusion.data.datasets.utils import ASPECT_RATIO_4096
from diffusion.data.datasets.sana_data_multi_scale import get_closest_ratio

Image.MAX_IMAGE_PIXELS = None

# ------------------------------
# Checkpoint Manager
# ------------------------------

class CheckpointManager:
    def __init__(self, checkpoint_dir: str, checkpoint_every: int = 1000):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # checkpoint paths
        self.progress_file = self.checkpoint_dir / "progress.json"
        self.processed_images_file = self.checkpoint_dir / "processed_images.pkl"
        self.samples_file = self.checkpoint_dir / "samples.pkl"
        self.config_file = self.checkpoint_dir / "config.json"
        
        # checkpoint data
        self.processed_images: Set[str] = set()
        self.saved_samples: List[dict] = []
        self.total_processed = 0
        self.last_checkpoint_time = time.time()
        
    def load_checkpoint(self):
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    self.total_processed = progress.get('total_processed', 0)
                    self.last_checkpoint_time = progress.get('last_checkpoint_time', time.time())
                    print(f"Loaded progress: {self.total_processed} samples processed")
            
            if self.processed_images_file.exists():
                with open(self.processed_images_file, 'rb') as f:
                    self.processed_images = pickle.load(f)
                    print(f"Loaded {len(self.processed_images)} processed image records")
            
            if self.samples_file.exists():
                with open(self.samples_file, 'rb') as f:
                    self.saved_samples = pickle.load(f)
                    print(f"Loaded {len(self.saved_samples)} saved samples")
            
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
    
    def save_checkpoint(self, new_samples: List[dict] = None):
        try:
            # add new samples if provided
            if new_samples:
                self.saved_samples.extend(new_samples)
                self.total_processed += len(new_samples)
                
                # record procesed images
                for sample in new_samples:
                    key = sample.get('__key__', '')
                    if key:
                        self.processed_images.add(key)
            
            # save progress
            progress = {
                'total_processed': self.total_processed,
                'last_checkpoint_time': time.time(),
                'samples_count': len(self.saved_samples)
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
            
            # save processed images
            with open(self.processed_images_file, 'wb') as f:
                pickle.dump(self.processed_images, f)
            
            # save sample data
            with open(self.samples_file, 'wb') as f:
                pickle.dump(self.saved_samples, f)
            
            self.last_checkpoint_time = time.time()
            print(f"Checkpoint saved: {self.total_processed} samples total")
            
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    def should_save_checkpoint(self):
        return self.total_processed % self.checkpoint_every == 0 and self.total_processed > 0
    
    def is_image_processed(self, image_key: str) -> bool:
        return image_key in self.processed_images
    
    def clear_checkpoints(self):
        try:
            if self.checkpoint_dir.exists():
                shutil.rmtree(self.checkpoint_dir)
                self.checkpoint_dir.mkdir(exist_ok=True)
            print("Checkpoints cleared")
        except Exception as e:
            print(f"Error clearing checkpoints: {e}")
    
    def save_config(self, config: dict):
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_saved_samples(self) -> List[dict]:
        return self.saved_samples.copy()

# ------------------------------
# CLI
# ------------------------------

def get_opts():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--img-dir", required=True)
    p.add_argument("--webdataset-dir", required=True)
    p.add_argument("--shard-size", type=int, default=1000)
    p.add_argument("--tile-size", type=int, default=2048)
    p.add_argument("--tile-overlap", type=int, default=256)
    p.add_argument("--scale", type=float, default=0.41407)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=max(os.cpu_count() // 2, 4))
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--text-batch-size", type=int, default=32, help="Text encoding batch size")
    p.add_argument("--checkpoint-every", type=int, default=1000, help="Save checkpoint every N samples")
    p.add_argument("--resume-from-checkpoint", action="store_true", help="Resume from last checkpoint")
    p.add_argument("--clear-checkpoints", action="store_true", help="Clear all checkpoints and restart")
    p.add_argument("--checkpoint-dir", default=None, help="Custom checkpoint directory")
    p.add_argument("--disable-compile", action="store_true", help="Disable torch.compile optimization")
    p.add_argument("--max-images", type=int, default=None, help="Limit number of images to process (for testing)")
    return p.parse_args()

# ------------------------------
# Dataset with Skip Support
# ------------------------------

class ImageCaptionDatasetWithSkip(Dataset):
    def __init__(self, img_dir: str, checkpoint_manager: CheckpointManager = None, target_dtype: torch.dtype = torch.float32, max_images: int = None):
        self.img_paths = sorted(
            p for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp")
            for p in Path(img_dir).rglob(ext)
        )
        
        # Limit images if specified (useful for testing)
        if max_images:
            self.img_paths = self.img_paths[:max_images]
            print(f"Limited to first {max_images} images for testing")
        
        self.checkpoint_manager = checkpoint_manager
        self.target_dtype = target_dtype
        
        # filter out already processed images
        if checkpoint_manager:
            original_count = len(self.img_paths)
            self.img_paths = [
                p for p in self.img_paths 
                if not checkpoint_manager.is_image_processed(p.stem)
            ]
            skipped_count = original_count - len(self.img_paths)
            if skipped_count > 0:
                print(f"Skipping {skipped_count} already processed images")
        
        print(f"Found {len(self.img_paths):,} images to process")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        stem = path.stem

        try:
            img = Image.open(path).convert("RGB")
            w0, h0 = img.size
            (h_t, w_t), _ = get_closest_ratio(h0, w0, ASPECT_RATIO_4096)
            h_t, w_t = int(h_t), int(w_t)

            # resize + center-crop to target
            if h_t / h0 > w_t / w0:
                img = img.resize((int(w0 * h_t / h0), h_t), Image.BICUBIC) 
            else:
                img = img.resize((w_t, int(h0 * w_t / w0)), Image.BICUBIC)
            left = (img.width - w_t) // 2
            top  = (img.height - h_t) // 2
            img = img.crop((left, top, left + w_t, top + h_t))

            tensor = TF.to_tensor(img).mul_(2).sub_(1).to(dtype=self.target_dtype)
            caption_path = path.with_suffix(".txt")
            caption = caption_path.read_text("utf-8").strip() if caption_path.exists() else ""

            return tensor, caption, w_t, h_t, stem
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            # Return a dummy tensor to avoid crashing the batch
            dummy_tensor = torch.zeros((3, 1024, 1024), dtype=self.target_dtype)
            return dummy_tensor, "", 1024, 1024, f"error_{idx}"

# ------------------------------
# Bucketed DataLoader
# ------------------------------

def build_bucketed_loader(dataset: ImageCaptionDatasetWithSkip, batch_size: int, num_workers: int):
    from torch.utils.data import DataLoader as TorchDataLoader

    print("Bucketing images by shape …")
    buckets: Dict[Tuple[int, int], List[int]] = {}

    for idx, p in enumerate(dataset.img_paths):
        try:
            with Image.open(p) as im:
                h0, w0 = im.height, im.width
            (h_t, w_t), _ = get_closest_ratio(h0, w0, ASPECT_RATIO_4096)
            key = (int(h_t), int(w_t))
            buckets.setdefault(key, []).append(idx)
        except Exception as e:
            print(f"Error bucketing {p}: {e}")
            continue

    # flatten into batch indices
    batches: List[List[int]] = []
    for inds in buckets.values():
        random.shuffle(inds)
        for i in range(0, len(inds), batch_size):
            batches.append(inds[i : i + batch_size])
    random.shuffle(batches)

    print(f"Created {len(batches)} batches from {len(buckets)} buckets")

    return TorchDataLoader(
        dataset,
        num_workers=num_workers,
        pin_memory=True,
        batch_sampler=batches,
    )

# ------------------------------
# Text Batch Queue
# ------------------------------

class TextBatchQueue:
    def __init__(self, max_batch_size: int, encode_fn):
        self.max_batch_size = max_batch_size
        self.encode_fn = encode_fn
        self.queue = []
    
    def add(self, caption: str, metadata: dict):
        self.queue.append((caption, metadata))
        
        if len(self.queue) >= self.max_batch_size:
            return self.flush()
        return []
    
    def flush(self):
        if not self.queue:
            return []
        
        captions = [item[0] for item in self.queue]
        metadatas = [item[1] for item in self.queue]
        
        try:
            embeddings = self.encode_fn(captions)
            results = []
            
            for embedding, metadata in zip(embeddings, metadatas):
                results.append((embedding, metadata))
            
            self.queue.clear()
            return results
            
        except Exception as e:
            print(f"Error in text batch processing: {e}")
            import traceback
            traceback.print_exc()
            self.queue.clear()
            return []

# ------------------------------
# Utility Functions for Tensor Conversion
# ------------------------------

def safe_tensor_to_numpy(tensor):
    if tensor.dtype == torch.bfloat16:
        return tensor.float().cpu().numpy().astype(np.float32)
    elif tensor.dtype == torch.float16:
        return tensor.float().cpu().numpy().astype(np.float32)
    else:
        return tensor.cpu().numpy().astype(np.float32)

def ensure_numpy_compatible(data):
    if isinstance(data, torch.Tensor):
        return safe_tensor_to_numpy(data)
    elif isinstance(data, np.ndarray):
        return data.astype(np.float32)
    else:
        return np.array(data, dtype=np.float32)

# ------------------------------
# Distributed Utilities
# ------------------------------

def setup_distributed():
    """Initialize distributed processing if running with multiple GPUs"""
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        if world_size > 1:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(local_rank)
            print(f"Initialized distributed processing: rank {local_rank}/{world_size}")
            return local_rank, world_size
    
    return 0, 1

def cleanup_distributed():
    """Clean up distributed processing"""
    if dist.is_initialized():
        dist.destroy_process_group()

def gather_samples_from_all_ranks(all_samples, webdataset_dir, world_size):
    """Gather samples from all ranks for final output"""
    if world_size == 1:
        return all_samples
    
    final_samples = list(all_samples)  # Start with current rank's samples
    
    # Collect from checkpoint files of other ranks
    for rank in range(world_size):
        if rank == dist.get_rank():  # Skip current rank
            continue
            
        other_checkpoint_dir = os.path.join(webdataset_dir, f".checkpoints_rank_{rank}")
        other_manager = CheckpointManager(other_checkpoint_dir, 1000)  # checkpoint_every doesn't matter for loading
        
        if other_manager.load_checkpoint():
            rank_samples = other_manager.get_saved_samples()
            final_samples.extend(rank_samples)
            print(f"Collected {len(rank_samples)} samples from rank {rank}")
    
    return final_samples

# ------------------------------
# Main
# ------------------------------

def main():
    opt = get_opts()
    
    # Setup distributed processing
    local_rank, world_size = setup_distributed()
    
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    
    print(f"Rank {local_rank}/{world_size} using device: {device}")
    
    # Create rank-specific checkpoint directory
    if opt.checkpoint_dir:
        checkpoint_dir = f"{opt.checkpoint_dir}_rank_{local_rank}"
    else:
        checkpoint_dir = os.path.join(opt.webdataset_dir, f".checkpoints_rank_{local_rank}")
    
    checkpoint_manager = CheckpointManager(checkpoint_dir, opt.checkpoint_every)
    
    # Handle checkpoint operations (only clear on rank 0 to avoid conflicts)
    if opt.clear_checkpoints and local_rank == 0:
        # Clear checkpoints for all ranks
        for rank in range(world_size):
            rank_checkpoint_dir = os.path.join(opt.webdataset_dir, f".checkpoints_rank_{rank}")
            rank_manager = CheckpointManager(rank_checkpoint_dir, opt.checkpoint_every)
            rank_manager.clear_checkpoints()
        print("All checkpoints cleared, starting fresh")
    elif opt.resume_from_checkpoint:
        if checkpoint_manager.load_checkpoint():
            print(f"Rank {local_rank} resumed from checkpoint")
        else:
            print(f"Rank {local_rank} found no checkpoint, starting fresh")
    
    # Synchronize all ranks before proceeding
    if world_size > 1:
        dist.barrier()
    
    # Save initial configuration (only on rank 0)
    if local_rank == 0:
        config = vars(opt)
        config['start_time'] = time.time()
        config['world_size'] = world_size
        checkpoint_manager.save_config(config)
    
    os.makedirs(opt.webdataset_dir, exist_ok=True)

    # Prepare model dtype
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    model_dtype = dtype_map[opt.precision]
    
    if opt.precision == "bf16" and not torch.cuda.is_bf16_supported():
        print("bfloat16 unsupported – falling back to fp32")
        model_dtype = torch.float32
        opt.precision = "fp32"
    elif opt.precision == "fp16" and not torch.cuda.is_available():
        print("CUDA not available – falling back to fp32")
        model_dtype = torch.float32
        opt.precision = "fp32"
    
    print(f"Using model dtype: {model_dtype}")
    torch.set_float32_matmul_precision("high")

    print("Loading DC-AE …")
    try:
        dae = AutoencoderDC.from_pretrained(
            "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
            torch_dtype=model_dtype
        ).to(device).eval()
        
        dae.enable_tiling(
            tile_sample_min_height=opt.tile_size,
            tile_sample_min_width=opt.tile_size,
            tile_sample_stride_height=opt.tile_overlap,
            tile_sample_stride_width=opt.tile_overlap
        )
        print(f"DAE loaded successfully with dtype: {next(dae.parameters()).dtype}")
    except Exception as e:
        print(f"Error loading DAE: {e}")
        raise
  
    print("Loading Gemma-2B-IT …")
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", trust_remote_code=True)
        text_encoder = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b-it",
            trust_remote_code=True,
            torch_dtype=model_dtype,
        ).to(device).eval()
        print(f"Text encoder loaded successfully with dtype: {next(text_encoder.parameters()).dtype}")
    except Exception as e:
        print(f"Error loading text encoder: {e}")
        raise

    # Optional torch.compile optimization
    if not opt.disable_compile:
        print("Attempting to compile DC-AE...")
        try:
            dae = torch.compile(dae, mode="default", fullgraph=False)
            print("DAE compilation successful")
        except Exception as e:
            print(f"DAE compile failed (continuing without compilation): {e}")
    else:
        print("Torch compile disabled")

    @torch.no_grad()
    def encode_latents(x):
        """Encoding latents with DC-AE"""
        try:
            x = x.to(device=device, dtype=model_dtype, non_blocking=True)
            latents = dae.encode(x).latent * opt.scale
            return latents.cpu()
        except RuntimeError as e:
            if "Input type" in str(e) and "should be the same" in str(e):
                print(f"Dtype mismatch detected, input: {x.dtype}, model: {next(dae.parameters()).dtype}")
                x = x.to(device=device, dtype=next(dae.parameters()).dtype, non_blocking=True)
                latents = dae.encode(x).latent * opt.scale
                return latents.cpu()
            else:
                raise e

    @torch.no_grad()
    def encode_caps(caps):
        if not caps or all(not c.strip() for c in caps):
            embed_dim = 2048 
            return torch.zeros((len(caps), embed_dim), dtype=torch.float32)
        
        try:
            toks = tokenizer(caps, return_tensors="pt", padding=True, truncation=True, max_length=512)
            toks = {k: v.to(device=device, non_blocking=True) for k, v in toks.items()}
            
            with torch.no_grad():
                outputs = text_encoder(**toks, output_hidden_states=True)
                hs = outputs.hidden_states[-1]  
                embeddings = hs.mean(dim=1)  
                
            return embeddings.cpu().float()  
            
        except Exception as e:
            print(f"Error encoding captions: {e}")
            embed_dim = 2048
            return torch.zeros((len(caps), embed_dim), dtype=torch.float32)

    def pack_npz(feat):
        buf = io.BytesIO()
        feat_array = ensure_numpy_compatible(feat)
        np.savez_compressed(
            buf,
            caption_feature=feat_array[None, None, :],
            attention_mask=np.ones((1, 1), np.int16)
        )
        buf.seek(0)
        return buf.getvalue()

    # Create dataset
    dataset = ImageCaptionDatasetWithSkip(opt.img_dir, checkpoint_manager, target_dtype=model_dtype, max_images=opt.max_images)
    
    # Split dataset across GPUs for distributed processing
    if len(dataset) > 0 and world_size > 1:
        images_per_gpu = len(dataset.img_paths) // world_size
        start_idx = local_rank * images_per_gpu
        if local_rank == world_size - 1:  # Last GPU gets remaining images
            end_idx = len(dataset.img_paths)
        else:
            end_idx = start_idx + images_per_gpu
        
        dataset.img_paths = dataset.img_paths[start_idx:end_idx]
        print(f"Rank {local_rank} processing {len(dataset.img_paths)} images ({start_idx}:{end_idx})")
    
    if len(dataset) == 0:
        print(f"Rank {local_rank}: All images have been processed!")
        all_samples = checkpoint_manager.get_saved_samples()
    else:
        loader = build_bucketed_loader(dataset, opt.batch_size, opt.num_workers)
        
        text_queue = TextBatchQueue(opt.text_batch_size, encode_caps)
        batch_samples = []
        start_time = time.time()
        
        print(f"Rank {local_rank}: Starting processing {len(dataset)} remaining images...")
        
        for batch_idx, (imgs, caps, ws, hs, stems) in enumerate(tqdm(loader, desc=f"Rank {local_rank}")):
            try:
                if batch_idx == 0:
                    print(f"Rank {local_rank} Batch {batch_idx}: input dtype={imgs.dtype}, model dtype={next(dae.parameters()).dtype}")
                
                lats = encode_latents(imgs)
                
                # Process each image in the batch
                for i, (cap, w, h, stem, lat) in enumerate(zip(caps, ws, hs, stems, lats)):
                    # Skip error samples
                    if stem.startswith("error_"):
                        continue
                        
                    metadata = {
                        'latent': lat,
                        'width': int(w),
                        'height': int(h),
                        'stem': stem,
                        'caption': cap
                    }
                    text_results = text_queue.add(cap, metadata)

                    for text_embedding, meta in text_results:
                        sample = create_sample(text_embedding, meta, pack_npz)
                        batch_samples.append(sample)
                
                # Save checkpoint periodically
                if len(batch_samples) >= opt.checkpoint_every:
                    checkpoint_manager.save_checkpoint(batch_samples)
                    batch_samples.clear()
                    elapsed = time.time() - start_time
                    rate = checkpoint_manager.total_processed / elapsed if elapsed > 0 else 0
                    print(f"Rank {local_rank} Progress: {checkpoint_manager.total_processed} samples, {rate:.1f}/sec")
                
            except Exception as e:
                print(f"Rank {local_rank} Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Process remaining text embeddings
        remaining_results = text_queue.flush()
        for text_embedding, metadata in remaining_results:
            sample = create_sample(text_embedding, metadata, pack_npz)
            batch_samples.append(sample)
        
        # Save final batch
        if batch_samples:
            checkpoint_manager.save_checkpoint(batch_samples)
        
        all_samples = checkpoint_manager.get_saved_samples()
        print(f"Rank {local_rank} completed processing {len(all_samples)} samples")

    # Synchronize all ranks before final output
    if world_size > 1:
        dist.barrier()
        print(f"Rank {local_rank} waiting for all ranks to complete...")

    # Only rank 0 writes the final output
    if local_rank == 0:
        print("Gathering results from all GPUs...")
        final_samples = gather_samples_from_all_ranks(all_samples, opt.webdataset_dir, world_size)
        
        print(f"Writing final WebDataset with {len(final_samples)} samples...")
        write_final_shards(final_samples, opt.webdataset_dir, opt.shard_size)
        
        # Calculate and print final statistics
        config = vars(opt)
        total_time = time.time() - config.get('start_time', time.time())
        print(f"\n=== Final Summary ===")
        print(f"Total samples: {len(final_samples)}")
        print(f"Total time: {total_time:.1f}s")
        if total_time > 0:
            print(f"Average rate: {len(final_samples)/total_time:.1f} samples/sec")
        print(f"Output directory: {opt.webdataset_dir}")
        print(f"Checkpoints saved in: {checkpoint_dir}")
    else:
        print(f"Rank {local_rank} finished processing, waiting for rank 0 to complete final output...")
    
    # Final synchronization
    if world_size > 1:
        dist.barrier()
        cleanup_distributed()
    
    print(f"Rank {local_rank} completed successfully!")

def create_sample(text_embedding, metadata, pack_npz_fn):
    """Create a sample dictionary for WebDataset"""
    buf_lat = io.BytesIO()
    latent_data = ensure_numpy_compatible(metadata['latent'])
    np.save(buf_lat, latent_data)
    buf_lat.seek(0)
    
    # JSON metadata for the training pipeline for each image
    meta_json = json.dumps({
        "file_name": f"{metadata['stem']}.jpg",
        "prompt": metadata['caption'],
        "width": metadata['width'],
        "height": metadata['height']
    }).encode()
    
    return {
        "__key__": metadata['stem'],
        "json": meta_json,
        "npy": buf_lat.getvalue(),
        "npz": pack_npz_fn(text_embedding)
    }

def write_final_shards(samples, output_dir, shard_size):
    if not samples:
        print("No samples to write!")
        return
        
    n_shards = math.ceil(len(samples) / shard_size)
    
    for sid in tqdm(range(n_shards), desc="Writing shards"):
        with wds.TarWriter(f"{output_dir}/shard_{sid:06d}.tar") as sink:
            start_idx = sid * shard_size
            end_idx = min(start_idx + shard_size, len(samples))
            for sample in samples[start_idx:end_idx]:
                sink.write(sample)
    
    # General metadata for the entire WebDataset
    meta = {
        "wids_version": 1,
        "name": "SANA_MS_precomputed",
        "description": "Latent & caption embeds (with distributed processing)",
        "shardlist": [
            {
                "url": f"shard_{i:06d}.tar", 
                "nsamples": min(shard_size, len(samples) - i * shard_size)
            }
            for i in range(n_shards)
        ]
    }
    
    with open(f"{output_dir}/wids-meta.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()