import os
import json
import requests
import pretty_midi
import numpy as np
import time
from typing import Optional, Dict, Any
from pathlib import Path

class OpenRouterMusicGenerator:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the music generator.
        
        Args:
            api_key: OpenRouter API key. If not provided, will attempt to get from
                    OPENROUTER_API_KEY environment variable.
        
        Raises:
            ValueError: If no API key is provided and OPENROUTER_API_KEY is not set.
        """
        # Try to get API key from parameter or environment
        self.api_key = api_key or os.environ.get('OPENROUTER_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. Either pass it to the constructor "
                "or set the OPENROUTER_API_KEY environment variable."
            )
        
        # Initialize HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.timeout = 30  # Default timeout in seconds
        self.max_retries = 3  # Maximum number of retries
        self.initial_backoff = 1  # Initial backoff time in seconds
    
    def _create_music_prompt(self, key: str, tempo: int, num_bars: int) -> str:
        """Create a prompt for the AI model to generate music."""
        return f"""Generate a {num_bars}-bar musical sequence in the key of {key} at {tempo} BPM.
        Return the sequence as a JSON array of MIDI notes, where each note is represented as:
        {{
            "pitch": <MIDI note number (0-127)>,
            "start_time": <start time in beats>,
            "duration": <duration in beats>,
            "velocity": <velocity (0-127)>
        }}
        Ensure the sequence follows musical theory and creates a coherent melody."""

    def _parse_ai_response(self, response):
        """Parse and validate the AI response.
        
        Args:
            response: The response object from the API request
            
        Returns:
            dict: Parsed music data
            
        Raises:
            ValueError: If the response is invalid or missing required fields
        """
        try:
            # Parse the response JSON
            response_data = response.json()
            
            # Extract the content from the response
            if 'choices' not in response_data or not response_data['choices']:
                raise ValueError("No choices in API response")
            
            content = response_data['choices'][0]['message']['content']
            
            # Log the raw content for debugging
            print(f"Raw API response content: {content}")
            
            # Try to clean the content before parsing
            content = content.strip()
            
            # Remove any markdown code block markers
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            
            content = content.strip()
            
            # Parse the content as JSON
            try:
                music_data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Content that failed to parse: {content}")
                raise ValueError(f"Invalid JSON in response: {str(e)}")
            
            # Validate required fields
            if 'notes' not in music_data:
                raise ValueError("Missing 'notes' field in response")
            
            # Validate each note
            for note in music_data['notes']:
                required_fields = ['pitch', 'start', 'end', 'velocity']
                for field in required_fields:
                    if field not in note:
                        raise ValueError(f"Note missing required field: {field}")
                
                # Validate note values
                if not (0 <= note['pitch'] <= 127):
                    raise ValueError(f"Invalid pitch value: {note['pitch']}")
                if not (0 <= note['velocity'] <= 127):
                    raise ValueError(f"Invalid velocity value: {note['velocity']}")
                if note['start'] >= note['end']:
                    raise ValueError(f"Invalid note timing: start ({note['start']}) >= end ({note['end']})")
            
            return music_data
            
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print(f"Response status code: {response.status_code}")
            print(f"Response headers: {response.headers}")
            raise ValueError(f"Failed to parse AI response: {str(e)}")

    def _make_api_request(self, 
                         messages: list, 
                         model: str, 
                         temperature: float = 0.7) -> Dict[str, Any]:
        """Make API request with retry mechanism and exponential backoff."""
        retry_count = 0
        backoff_time = self.initial_backoff
        
        while retry_count <= self.max_retries:
            try:
                response = self.session.post(
                    self.base_url,
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": temperature
                    },
                    timeout=self.timeout
                )
                
                # Check for successful response
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                if retry_count == self.max_retries:
                    raise Exception("API request timed out after multiple retries")
                print(f"Request timed out. Retrying in {backoff_time} seconds...")
                
            except requests.exceptions.RequestException as e:
                if retry_count == self.max_retries:
                    raise Exception(f"API request failed after {self.max_retries} retries: {str(e)}")
                print(f"Request failed: {str(e)}. Retrying in {backoff_time} seconds...")
            
            # Wait before retrying
            time.sleep(backoff_time)
            retry_count += 1
            backoff_time *= 2  # Exponential backoff
        
        raise Exception("Maximum retry attempts reached")

    def generate_midi(self, prompt: str, tempo: int = 120, key: str = "C", 
                     model_id: str = "anthropic/claude-3-opus-20240229") -> pretty_midi.PrettyMIDI:
        """Generate MIDI music based on a text prompt.
        
        Args:
            prompt: Text description of the music to generate
            tempo: Tempo in BPM (default: 120)
            key: Musical key (default: C)
            model_id: OpenRouter model ID to use
            
        Returns:
            pretty_midi.PrettyMIDI object containing the generated music
            
        Raises:
            ValueError: If the prompt is empty or invalid
            RuntimeError: If the API request fails
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Prepare the system prompt with music theory context
        system_prompt = (
            "You are a music composition expert. Your response must be a valid JSON object with the following structure:\n"
            "{\n"
            "  'notes': [\n"
            "    {\n"
            "      'pitch': int,  # MIDI note number (0-127)\n"
            "      'start': float,  # Start time in seconds\n"
            "      'end': float,  # End time in seconds\n"
            "      'velocity': int  # Note velocity (0-127)\n"
            "    },\n"
            "    ...\n"
            "  ],\n"
            "  'tempo': int,  # Tempo in BPM\n"
            "  'key': str  # Musical key\n"
            "}\n"
            "Do not include any text before or after the JSON object. The response must be a single valid JSON object.\n"
            f"Generate music in {key} at {tempo} BPM based on this description: {prompt}"
        )
        
        # Prepare the API request
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                response = self.session.post(
                    self.base_url,
                    json={
                        "model": model_id,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 2000
                    },
                    timeout=self.timeout
                )
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Parse the response using the safer method
                try:
                    music_data = self._parse_ai_response(response)
                except ValueError as e:
                    raise RuntimeError(f"Failed to parse AI response: {str(e)}")
                
                # Create MIDI file
                midi_data = pretty_midi.PrettyMIDI()
                piano_program = pretty_midi.Instrument(program=0)  # Piano
                
                # Add notes to the instrument
                for note_data in music_data['notes']:
                    note = pretty_midi.Note(
                        velocity=note_data['velocity'],
                        pitch=note_data['pitch'],
                        start=note_data['start'],
                        end=note_data['end']
                    )
                    piano_program.notes.append(note)
                
                # Add the instrument to the MIDI file
                midi_data.instruments.append(piano_program)
                
                # Set the tempo using the correct attribute
                midi_data.tempo = tempo
                
                return midi_data
                
            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count > self.max_retries:
                    raise RuntimeError(f"API request failed after {self.max_retries} retries: {str(e)}")
                time.sleep(1)  # Wait before retrying

    def save_midi(self, midi_data: pretty_midi.PrettyMIDI, filepath: str) -> None:
        """Save MIDI data to a file."""
        try:
            midi_data.write(filepath)
        except Exception as e:
            raise Exception(f"Failed to save MIDI file: {str(e)}") 