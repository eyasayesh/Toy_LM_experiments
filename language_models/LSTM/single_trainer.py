import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
import traceback

# Define a custom TokenDataset class directly in this file to avoid import issues
class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer_fn, seq_len):
        """
        Args:
            text (str): Raw input text
            tokenizer_fn (callable): Function that maps text to list of token IDs
            seq_len (int): Sequence length
        """
        self.seq_len = seq_len

        # Tokenize the entire corpus at once
        self.tokens = tokenizer_fn(text)
        self.tokens = [t for t in self.tokens if t != -1]  # filter out any unknowns if needed
        print(f"Tokenized text. Total tokens: {len(self.tokens)}")

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.seq_len]
        y = self.tokens[idx + 1:idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Define the LSTM model directly in this file to avoid import issues
class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=1):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        return self.linear(out)

def get_tokenizer(tokenizer_type, tokenizer_loc):
    """
    Dummy tokenizer function to avoid import errors.
    In a real scenario, this would load the actual tokenizer.
    """
    print(f"Loading tokenizer of type {tokenizer_type} from {tokenizer_loc}")
    
    # First check if the tokenizer file exists
    if not os.path.exists(tokenizer_loc):
        print(f"WARNING: Tokenizer file {tokenizer_loc} does not exist!")
        # Return a dummy tokenizer and a small vocab size
        return lambda text: [0] * min(len(text), 1000), 10
    
    # Mock implementation - would be replaced with actual tokenizer code
    # For debugging, let's create a very simple character tokenizer
    
    try:
        # Try to import custom tokenizers if they exist
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        if tokenizer_type == "char":
            from custom_tokenizers import CharTokenizer
            tokenizer = CharTokenizer()
            tokenizer.load(tokenizer_loc)
            vocab_size = len(tokenizer.char2id)
            return tokenizer.encode, vocab_size
        else:
            from custom_tokenizers import WordTokenizer, BPETokenizer, UniformPairTokenizer, FreqWeightedTokenizer
            if tokenizer_type == "word":
                tokenizer = WordTokenizer()
            elif tokenizer_type == "bpe":
                tokenizer = BPETokenizer()
            elif tokenizer_type == "uniform":
                tokenizer = UniformPairTokenizer()
            elif tokenizer_type == "freq":
                tokenizer = FreqWeightedTokenizer()
            else:
                raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
            
            tokenizer.load(tokenizer_loc)
            vocab_size = len(tokenizer.vocab)
            return tokenizer.encode, vocab_size
    except ImportError as e:
        print(f"WARNING: Could not import custom tokenizers: {e}")
        print("Using a simple character tokenizer instead")
        
        # Simple character level tokenizer
        chars = set()
        with open(tokenizer_loc, 'r') as f:
            sample = f.read(10000)  # Read a sample to get character set
            chars = set(sample)
        
        char2id = {c: i+1 for i, c in enumerate(sorted(chars))}
        char2id['<unk>'] = 0
        
        vocab_size = len(char2id)
        print(f"Created simple char tokenizer with vocab size: {vocab_size}")
        
        def encode(text):
            return [char2id.get(c, 0) for c in text]
        
        return encode, vocab_size

def train(text, tokenizer_fn, vocab_size, seq_len, batch_size, epochs, lr, tokenizer_name, use_wandb=False):
    """Train an LSTM model on a single GPU."""
    # Print diagnostic information
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'None'}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        print(f"Using device: {device} - {torch.cuda.get_device_name(0)}")
    
    try:
        # Initialize model and move to device
        print(f"Creating LSTM model with vocab size: {vocab_size}")
        model = LSTMModel(vocab_size).to(device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Create dataset
        print(f"Creating dataset with sequence length: {seq_len}")
        dataset = TokenDataset(text, tokenizer_fn, seq_len)
        
        if len(dataset) == 0:
            print("ERROR: Dataset is empty! Check your tokenizer and text data.")
            return None
        
        print(f"Dataset size: {len(dataset)} samples")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=0  # Set to 0 for debugging
        )
        
        # Initialize wandb if requested
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project="lstm-language-model",
                    config={
                        "tokenizer": tokenizer_name,
                        "vocab_size": vocab_size,
                        "seq_len": seq_len,
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "lr": lr,
                        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "None",
                        "pytorch_version": torch.__version__
                    }
                )
            except ImportError:
                print("wandb not installed. Logging disabled.")
                use_wandb = False
        
        # Training loop
        print(f"Starting training for {epochs} epochs")
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            batch_count = 0
            
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, vocab_size), targets.view(-1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                if batch_idx % 10 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {avg_loss:.4f}")
                    if use_wandb:
                        wandb.log({"loss": avg_loss})
            
            # Log epoch results
            if batch_count > 0:
                avg_epoch_loss = total_loss / batch_count
                print(f"Epoch {epoch} completed: Average Loss = {avg_epoch_loss:.4f}")
                if use_wandb:
                    wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch})
            else:
                print(f"Epoch {epoch} had no batches!")
        
        # Save model
        save_path = f"checkpoints/lstm_{tokenizer_name}.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
        return model
    
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        return None

def main():
    # Print CUDA and GPU information
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'None'}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True,
                      choices=["char", "word", "bpe", "uniform", "freq"],
                      help="Type of tokenizer to use")
    parser.add_argument("--tokenizer_loc", type=str, required=True,
                      help="Full path to the tokenizer file")
    parser.add_argument("--data_path", type=str, default="data/text8.txt")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tokenizer_name", type=str, default="default")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"ERROR: Data file {args.data_path} does not exist!")
        return
    
    print(f"Reading text from {args.data_path}")
    try:
        with open(args.data_path, "r") as f:
            text = f.read()
            # Optional truncation for faster testing
            text = text[:100_000]
            print(f"Read {len(text)} characters from data file")
    except Exception as e:
        print(f"Error reading data file: {e}")
        return

    print(f"Getting tokenizer of type '{args.tokenizer}' from {args.tokenizer_loc}")
    tokenizer_fn, vocab_size = get_tokenizer(args.tokenizer, args.tokenizer_loc)

    print(f"Training on single GPU using '{args.tokenizer}' tokenizer")
    print(f"Vocabulary size: {vocab_size}")

    train(
        text,
        tokenizer_fn,
        vocab_size,
        args.seq_len,
        args.batch_size,
        args.epochs,
        args.lr,
        args.tokenizer_name,
        use_wandb=not args.no_wandb
    )

if __name__ == "__main__":
    main()