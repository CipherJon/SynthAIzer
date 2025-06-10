import torch
import os
import argparse
from synthaizer.model import MusicGenerator
from pathlib import Path

def train_model(
    midi_dir: str,
    output_dir: str,
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    seq_length: int = 512
):
    """
    Train the music generation model.
    
    Args:
        midi_dir: Directory containing MIDI files for training
        output_dir: Directory to save model checkpoints
        batch_size: Number of samples per training batch
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        seq_length: Maximum sequence length for training
    """
    try:
        # Validate directories
        if not os.path.exists(midi_dir):
            raise ValueError(f"MIDI directory not found: {midi_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate parameters
        if num_epochs < 1:
            raise ValueError("Number of epochs must be positive")
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if seq_length < 32:
            raise ValueError("Sequence length must be at least 32")
        
        print("Initializing model and dataset...")
        try:
            # Create dataset
            dataset = MIDIDataset(midi_dir, seq_length=seq_length)
            if len(dataset) == 0:
                raise ValueError("No valid MIDI files found in directory")
            
            # Create data loader
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Initialize model
            model = MusicTransformer(
                vocab_size=128,  # MIDI note range
                embedding_dim=256,
                num_heads=8,
                num_layers=6,
                seq_length=seq_length
            )
            
            # Move model to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            print(f"Using device: {device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        print(f"Starting training for {num_epochs} epochs...")
        try:
            # Training loop
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0
                num_batches = 0
                
                for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
                    try:
                        # Move data to device
                        input_seq = input_seq.to(device)
                        target_seq = target_seq.to(device)
                        
                        # Forward pass
                        optimizer.zero_grad()
                        output = model(input_seq)
                        
                        # Reshape for loss calculation
                        output = output.view(-1, output.size(-1))
                        target = target_seq.view(-1)
                        
                        # Calculate loss
                        loss = criterion(output, target)
                        
                        # Backward pass
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                        num_batches += 1
                        
                        # Print progress
                        if (batch_idx + 1) % 10 == 0:
                            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
                            
                    except Exception as e:
                        print(f"Error in batch {batch_idx}: {str(e)}")
                        continue
                
                # Print epoch summary
                avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
                print(f"Epoch {epoch+1}/{num_epochs} completed. Average loss: {avg_loss:.4f}")
                
                # Save checkpoint
                try:
                    checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pt")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                    }, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
                except Exception as e:
                    print(f"Failed to save checkpoint: {str(e)}")
            
            # Save final model
            try:
                final_model_path = os.path.join(output_dir, "model_final.pt")
                torch.save(model.state_dict(), final_model_path)
                print(f"Training completed. Final model saved to {final_model_path}")
            except Exception as e:
                print(f"Failed to save final model: {str(e)}")
            
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the music generation model")
    parser.add_argument("--midi_dir", type=str, required=True,
                      help="Directory containing MIDI files for training")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Number of samples per training batch")
    parser.add_argument("--num_epochs", type=int, default=10,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                      help="Learning rate for optimizer")
    parser.add_argument("--seq_length", type=int, default=512,
                      help="Maximum sequence length for training")
    
    args = parser.parse_args()
    
    train_model(
        midi_dir=args.midi_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        seq_length=args.seq_length
    ) 