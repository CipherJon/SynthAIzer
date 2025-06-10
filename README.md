# SynthAIzer - AI-Powered Music Generator for LMMS

SynthAIzer is an AI-powered music generation tool that creates MIDI patterns compatible with LMMS (Linux MultiMedia Studio). It uses OpenRouter's API to access state-of-the-art AI models for generating musical patterns that can be imported into LMMS for further editing and production.

## Features

- AI-powered music generation using OpenRouter's API
- Support for multiple AI models (Claude, GPT-4, Gemini)
- MIDI file export compatible with LMMS
- Customizable generation parameters (tempo, key, style)
- Real-time preview capabilities
- Easy integration with LMMS workflow
- Persistent configuration storage
- Automatic LMMS path detection

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/SynthAIzer.git
cd SynthAIzer
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Get an API key from OpenRouter:
   - Visit [OpenRouter](https://openrouter.ai/)
   - Create an account and get your API key
   - The key will be saved securely in the configuration

## Usage

1. Run the GUI:
```bash
python run_gui.py
```

2. First-time setup:
   - Enter your OpenRouter API key in the top section
   - Set your LMMS path (the program will try to find it automatically)
   - Test the LMMS connection using the "Test LMMS" button

3. Generate music:
   - Select your preferred AI model
   - Set generation parameters (bars, tempo, key, creativity)
   - Click "Generate Music"
   - Use the control buttons to:
     - Export to MIDI
     - Import into LMMS
     - Preview in LMMS

## Configuration

The program saves your settings in a configuration file:
- Windows: `C:\Users\YourUsername\.synthaizer\config.json`
- Linux/macOS: `~/.synthaizer/config.json`

Saved settings include:
- OpenRouter API key
- LMMS installation path
- Last used AI model
- Generation parameters

## Supported AI Models

- Claude 3 Opus (anthropic/claude-3-opus-20240229)
- Claude 3 Sonnet (anthropic/claude-3-sonnet-20240229)
- Google Gemini Pro (google/gemini-pro)
- GPT-4 Turbo (openai/gpt-4-turbo-preview)

## Requirements

- Python 3.8 or higher
- LMMS installed on your system
- OpenRouter API key
- Internet connection for AI model access

## Troubleshooting

1. **LMMS Not Found**:
   - Use the "Browse" button to locate your LMMS executable
   - Common Windows paths:
     - `C:\Program Files\LMMS\lmms.exe`
     - `C:\Program Files (x86)\LMMS\lmms.exe`
     - `%LOCALAPPDATA%\LMMS\lmms.exe`

2. **API Key Issues**:
   - Verify your OpenRouter API key is correct
   - Check your internet connection
   - Try a different AI model if one fails

3. **MIDI Export Problems**:
   - Ensure you have write permissions in the target directory
   - Check if LMMS is properly configured for MIDI import

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE V2 - see the LICENSE file for details.
