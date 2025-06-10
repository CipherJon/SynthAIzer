import argparse
import os
import sys
from pathlib import Path
from .openrouter_model import OpenRouterMusicGenerator
from .config import ConfigManager

def validate_temperature(value: str) -> float:
    """Validate that temperature is between 0.1 and 1.0."""
    try:
        temp = float(value)
        if not 0.1 <= temp <= 1.0:
            raise argparse.ArgumentTypeError(
                f"Temperature must be between 0.1 and 1.0, got {temp}"
            )
        return temp
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Temperature must be a number between 0.1 and 1.0, got {value}"
        )

def main():
    parser = argparse.ArgumentParser(description="Generate music using SynthAIzer")
    
    # API configuration
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenRouter API key (default: from environment or config)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Description of the music to generate"
    )
    parser.add_argument(
        "--tempo",
        type=int,
        default=120,
        help="Tempo in BPM (default: 120)"
    )
    parser.add_argument(
        "--key",
        type=str,
        default="C",
        help="Musical key (default: C)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-3-opus-20240229",
        help="AI model to use for generation"
    )
    parser.add_argument(
        "--temperature",
        type=validate_temperature,
        default=0.7,
        help="Temperature for generation (0.1 to 1.0, default: 0.7)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: from config)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Output filename (default: based on prompt)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration
        config = ConfigManager()
        
        # Get API key from args, environment, or config
        api_key = args.api_key or os.getenv("OPENROUTER_API_KEY") or config.get_api_key()
        if not api_key:
            parser.error("API key is required. Set it with --api-key, OPENROUTER_API_KEY environment variable, or in config.")
        
        # Initialize generator with error handling
        try:
            print("Initializing music generator...")
            generator = OpenRouterMusicGenerator(api_key)
        except Exception as e:
            print(f"Error initializing music generator: {str(e)}")
            print("Please check your API key and internet connection.")
            return 1
        
        try:
            # Generate music with progress feedback
            print(f"Generating music with prompt: {args.prompt}")
            print(f"Using model: {args.model}")
            print(f"Tempo: {args.tempo} BPM, Key: {args.key}")
            
            midi_data = generator.generate_midi(
                prompt=args.prompt,
                tempo=args.tempo,
                key=args.key,
                model_id=args.model
            )
            
            # Determine output path
            output_dir = args.output_dir or config.get_output_dir()
            output_name = args.output_name or f"{args.prompt[:30].replace(' ', '_')}.mid"
            output_path = os.path.join(output_dir, output_name)
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Save MIDI file with error handling
            try:
                midi_data.write(output_path)
                print(f"Music generated and saved to: {output_path}")
            except Exception as e:
                print(f"Error saving MIDI file: {str(e)}")
                print(f"Please check if you have write permissions in: {output_dir}")
                return 1
            
        except Exception as e:
            print(f"Error generating music: {str(e)}")
            print("This might be due to:")
            print("- Invalid generation parameters")
            print("- API rate limiting")
            print("- Network connectivity issues")
            print("- Model availability")
            return 1
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print("Please report this error to the developers.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 