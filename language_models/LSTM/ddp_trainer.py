import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from dataset import TokenDataset
from model import LSTMModel
import wandb
import os

def train(rank, world_size, text, tokenizer_fn, vocab_size, seq_len, batch_size, epochs, lr, tokenizer_name):
    # Print diagnostic information
    print(f"Process {rank}: CUDA available: {torch.cuda.is_available()}")
    print(f"Process {rank}: CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'None'}")
    print(f"Process {rank}: PyTorch version: {torch.__version__}")
    print(f"Process {rank}: CUDA device count: {torch.cuda.device_count()}")
    
    # Initialize process group
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU configuration and PyTorch installation.")
    
    try:
        # Initialize process group with NCCL backend
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            world_size=world_size,
            rank=rank
        )
        
        # Set device for this process
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        print(f"Process {rank}: Using device: {device}")
        
        # Initialize model and move to device
        model = LSTMModel(vocab_size).to(device)
        model = DDP(model, device_ids=[rank])
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Create dataset and sampler
        dataset = TokenDataset(text, tokenizer_fn, seq_len)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=4
        )
        
        # Initialize wandb
        if rank == 0:
            wandb.init(
                project="lstm-language-model",
                config={
                    "tokenizer": tokenizer_name,
                    "vocab_size": vocab_size,
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "lr": lr,
                    "world_size": world_size,
                    "cuda_version": torch.version.cuda,
                    "pytorch_version": torch.__version__
                }
            )
        
        # Training loop
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            model.train()
            total_loss = 0
            
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, vocab_size), targets.view(-1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if rank == 0 and batch_idx % 100 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {avg_loss:.4f}")
                    wandb.log({"loss": avg_loss})
    
    except Exception as e:
        print(f"Error in process {rank}: {str(e)}")
        raise e
    
    finally:
        # Cleanup
        dist.destroy_process_group()
