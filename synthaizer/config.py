import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

class ConfigManager:
    def __init__(self):
        self.config_dir = Path.home() / '.synthaizer'
        self.config_file = self.config_dir / 'config.json'
        
        # Load config first
        self.config = self._load_config()
        
        # If defaults were added, save the config
        if not self.config_file.exists():
            self.save_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default if not exists."""
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        default_config = {
            'api_key': '',
            'lmms_path': '',
            'last_model': 'anthropic/claude-3-opus-20240229',
            'output_dir': str(Path.home() / 'Music' / 'SynthAIzer')
        }
        
        # Load existing config or use defaults
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Ensure all default keys exist
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except (json.JSONDecodeError, IOError):
                return default_config
        else:
            return default_config
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except IOError as e:
            print(f"Error saving config: {e}")
    
    def get_api_key(self) -> str:
        """Get the OpenRouter API key from config or environment variable."""
        # First try to get from config
        api_key = self.config.get('api_key', '')
        
        # If not found or empty in config, try environment variable
        if not api_key:
            api_key = os.environ.get('OPENROUTER_API_KEY', '')
            
        return api_key
    
    def set_api_key(self, api_key: str):
        """Set the API key in config."""
        self.config['api_key'] = api_key
        self.save_config()
    
    def get_lmms_path(self) -> str:
        """Get the LMMS path from config."""
        return self.config.get('lmms_path', '')
    
    def set_lmms_path(self, path: str):
        """Set the LMMS path in config."""
        self.config['lmms_path'] = path
        self.save_config()
    
    def get_last_model(self) -> str:
        """Get the last used model from config."""
        return self.config.get('last_model', 'anthropic/claude-3-opus-20240229')
    
    def set_last_model(self, model: str):
        """Set the last used model in config."""
        self.config['last_model'] = model
        self.save_config()
    
    def get_output_dir(self) -> str:
        """Get the output directory from config."""
        output_dir = Path(self.config.get('output_dir', str(Path.home() / 'Music' / 'SynthAIzer')))
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)
    
    def set_output_dir(self, path: str):
        """Set the output directory in config."""
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.config['output_dir'] = str(output_dir)
        self.save_config() 