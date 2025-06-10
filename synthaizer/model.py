import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pretty_midi
import os
from typing import List, Tuple, Optional
import numpy as np
import math

class MIDIDataset(Dataset):
    """Dataset for MIDI note sequences."""
    def __init__(self, midi_files, seq_length=32):
        self.seq_length = seq_length
        self.notes = []
        
        # Process each MIDI file
        for midi_file in midi_files:
            try:
                midi_data = pretty_midi.PrettyMIDI(midi_file)
                for instrument in midi_data.instruments:
                    for note in instrument.notes:
                        self.notes.append(note.pitch)
            except Exception as e:
                print(f"Error processing {midi_file}: {str(e)}")
                continue
    
    def __len__(self):
        return len(self.notes) - self.seq_length
    
    def __getitem__(self, idx):
        # Get sequence of notes
        notes = self.notes[idx:idx + self.seq_length + 1]
        
        # Create input and target tensors
        input_tensor = torch.tensor(notes[:-1], dtype=torch.long)
        target_tensor = torch.tensor(notes[1:], dtype=torch.long)
        
        return input_tensor, target_tensor

class MusicTransformer(nn.Module):
    """Transformer model for music generation."""
    def __init__(self, vocab_size: int, embedding_dim: int = 256, num_heads: int = 8, 
                 num_layers: int = 6, seq_length: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding
        self.register_buffer(
            "positional_encoding",
            self._create_sinusoidal_encoding(seq_length, embedding_dim)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
    
    def _create_sinusoidal_encoding(self, seq_length: int, embedding_dim: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        position = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        
        pos_encoding = torch.zeros(1, seq_length, embedding_dim)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_length)
        batch_size = x.size(0)
        
        # Get token embeddings
        # Shape: (batch_size, seq_length, embedding_dim)
        token_embeddings = self.token_embedding(x)
        
        # Add positional encoding
        # Shape: (batch_size, seq_length, embedding_dim)
        embeddings = token_embeddings + self.positional_encoding[:, :x.size(1), :]
        
        # Create attention mask for padding
        # Shape: (batch_size, seq_length)
        mask = (x != 0).unsqueeze(1).unsqueeze(2)
        
        # Apply transformer
        # Shape: (batch_size, seq_length, embedding_dim)
        transformer_output = self.transformer(embeddings, src_key_padding_mask=~mask.squeeze(1))
        
        # Project to vocabulary
        # Shape: (batch_size, seq_length, vocab_size)
        output = self.output_layer(transformer_output)
        
        return output

class MusicGenerator(nn.Module):
    def __init__(self, model_name: str = "gpt2", seq_length: int = 512):
        super().__init__()
        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1
            ),
            num_layers=6
        )
        self.embedding = nn.Embedding(128, 512)  # 128 possible MIDI notes
        self.fc = nn.Linear(512, 128)  # Output layer for note prediction
        self.seq_length = seq_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_length)
        x = self.embedding(x)  # (batch_size, seq_length, 512)
        x = x.permute(1, 0, 2)  # (seq_length, batch_size, 512)
        x = self.model(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_length, 512)
        x = self.fc(x)  # (batch_size, seq_length, 128)
        return x
    
    def train_model(self, 
                   midi_dir: str,
                   batch_size: int = 32,
                   num_epochs: int = 10,
                   learning_rate: float = 1e-4,
                   save_path: Optional[str] = None) -> None:
        """Train the model on MIDI files."""
        # Create dataset and dataloader
        dataset = MIDIDataset(midi_dir, self.seq_length)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                # Move tensors to device
                inputs = inputs.to(self.device)  # (batch_size, seq_length)
                targets = targets.to(self.device)  # (batch_size, seq_length)
                
                # Forward pass
                outputs = self(inputs)  # (batch_size, seq_length, 128)
                
                # Reshape for CrossEntropyLoss
                outputs = outputs.view(-1, 128)  # (batch_size * seq_length, 128)
                targets = targets.view(-1)  # (batch_size * seq_length)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Print progress
                if (batch_idx + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], '
                          f'Batch [{batch_idx+1}/{len(dataloader)}], '
                          f'Loss: {loss.item():.4f}')
            
            # Print epoch statistics
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
            
            # Save model if path is provided
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, os.path.join(save_path, f'model_epoch_{epoch+1}.pt'))
    
    def generate_midi(self, 
                     prompt: str,
                     tempo: int = 120,
                     key: str = 'C',
                     model_id: str = 'anthropic/claude-3-opus-20240229') -> pretty_midi.PrettyMIDI:
        """Generate MIDI music using the OpenRouter API."""
        # ... rest of the method remains unchanged ...

    def _get_midi_note(self, note: str, octave: int) -> int:
        """Convert a note name to MIDI note number."""
        # Map of note names to their position in the scale (0-11)
        key_map = {
            'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
            'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
        }
        
        # Convert to uppercase and remove any whitespace
        note = note.strip().upper()
        
        # Get the note number, defaulting to C if not found
        note_number = key_map.get(note)
        if note_number is None:
            raise ValueError(
                f"Invalid note name: {note}. "
                f"Expected one of: {', '.join(key_map.keys())}"
            )
        
        # Calculate MIDI note number (C4 = 60)
        return note_number + (octave + 1) * 12

    def _get_key_notes(self, key: str) -> List[int]:
        """Get the notes in a given key."""
        # Map of key names to their scale degrees (0-11)
        key_map = {
            'C': [0, 2, 4, 5, 7, 9, 11],  # C major
            'G': [7, 9, 11, 0, 2, 4, 6],  # G major
            'D': [2, 4, 6, 7, 9, 11, 1],  # D major
            'A': [9, 11, 1, 2, 4, 6, 8],  # A major
            'E': [4, 6, 8, 9, 11, 1, 3],  # E major
            'B': [11, 1, 3, 4, 6, 8, 10], # B major
            'F#': [6, 8, 10, 11, 1, 3, 5], # F# major
            'C#': [1, 3, 5, 6, 8, 10, 0], # C# major
            'F': [5, 7, 9, 10, 0, 2, 4],  # F major
            'Bb': [10, 0, 2, 3, 5, 7, 9], # Bb major
            'Eb': [3, 5, 7, 8, 10, 0, 2], # Eb major
            'Ab': [8, 10, 0, 1, 3, 5, 7], # Ab major
            'Db': [1, 3, 5, 6, 8, 10, 0], # Db major
            'Gb': [6, 8, 10, 11, 1, 3, 5] # Gb major
        }
        
        # Convert to uppercase and remove any whitespace
        key = key.strip().upper()
        
        # Get the scale degrees, defaulting to C major if not found
        scale_degrees = key_map.get(key)
        if scale_degrees is None:
            raise ValueError(
                f"Invalid key: {key}. "
                f"Expected one of: {', '.join(key_map.keys())}"
            )
        
        # Convert scale degrees to MIDI note numbers (C4 = 60)
        return [degree + 60 for degree in scale_degrees]

    def _get_chord_notes(self, root: str, chord_type: str) -> List[int]:
        """Get the notes in a chord."""
        # Map of chord types to their intervals from the root
        chord_map = {
            'major': [0, 4, 7],
            'minor': [0, 3, 7],
            'diminished': [0, 3, 6],
            'augmented': [0, 4, 8],
            'major7': [0, 4, 7, 11],
            'minor7': [0, 3, 7, 10],
            'dominant7': [0, 4, 7, 10],
            'diminished7': [0, 3, 6, 9],
            'half_diminished7': [0, 3, 6, 10],
            'major6': [0, 4, 7, 9],
            'minor6': [0, 3, 7, 9],
            'major9': [0, 4, 7, 11, 14],
            'minor9': [0, 3, 7, 10, 14],
            'dominant9': [0, 4, 7, 10, 14],
            'major11': [0, 4, 7, 11, 14, 17],
            'minor11': [0, 3, 7, 10, 14, 17],
            'dominant11': [0, 4, 7, 10, 14, 17],
            'major13': [0, 4, 7, 11, 14, 17, 21],
            'minor13': [0, 3, 7, 10, 14, 17, 21],
            'dominant13': [0, 4, 7, 10, 14, 17, 21]
        }
        
        # Convert to lowercase and remove any whitespace
        chord_type = chord_type.strip().lower()
        
        # Get the chord intervals, defaulting to major if not found
        intervals = chord_map.get(chord_type)
        if intervals is None:
            raise ValueError(
                f"Invalid chord type: {chord_type}. "
                f"Expected one of: {', '.join(chord_map.keys())}"
            )
        
        # Get the root note's MIDI number
        root_note = self._get_midi_note(root, 4)  # Use octave 4 as base
        
        # Calculate the chord notes
        return [root_note + interval for interval in intervals]

    def save_midi(self, midi_data, filename):
        """
        Save MIDI data to file
        """
        midi_data.write(filename) 