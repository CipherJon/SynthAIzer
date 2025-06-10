import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import subprocess
from .openrouter_model import OpenRouterMusicGenerator
from .config import ConfigManager
from pathlib import Path
import threading
import queue
import sys

class SynthAIzerGUI:
    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("SynthAIzer")
        self.root.geometry("800x600")
        
        # Create main scrollable frame
        self.main_canvas = tk.Canvas(self.root)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        self.main_frame = ttk.Frame(self.main_canvas)
        
        # Configure canvas scrolling
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.main_canvas.bind('<Configure>', lambda e: self.main_canvas.configure(
            scrollregion=self.main_canvas.bbox("all")))
        
        # Create window in canvas for the main frame
        self.main_canvas.create_window((0, 0), window=self.main_frame, anchor="nw")
        
        # Pack the canvas and scrollbar
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to scrolling
        self.main_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Initialize configuration
        self.config = ConfigManager()
        
        # Initialize variables
        self.generator = None
        self.api_key = self.config.get_api_key()
        self.current_midi = None
        
        # Default LMMS paths for different operating systems
        self.default_lmms_paths = {
            'nt': [
                r'C:\Program Files\LMMS\lmms.exe',
                r'C:\Program Files (x86)\LMMS\lmms.exe',
                os.path.expanduser('~\\AppData\\Local\\LMMS\\lmms.exe'),
                r'C:\Program Files\LMMS\bin\lmms.exe',
                r'C:\Program Files (x86)\LMMS\bin\lmms.exe'
            ],
            'posix': [
                '/usr/bin/lmms',
                '/usr/local/bin/lmms',
                os.path.expanduser('~/lmms/lmms'),
                '/opt/lmms/bin/lmms'
            ]
        }
        
        # Initialize LMMS path before creating GUI
        self.lmms_path = self.config.get_lmms_path()
        if not self.lmms_path:
            # Create a temporary status text for LMMS detection
            temp_status = tk.Text(self.root, height=1, width=1)
            temp_status.pack_forget()  # Hide it
            self.status_text = temp_status
            self.lmms_path = self.find_lmms()
            self.status_text = None  # Clear the temporary status text
        
        # Create GUI sections
        self.create_api_section()
        self.create_lmms_section()
        self.create_generation_section()
        self.create_controls_section()
        self.create_status_section()
        
        # Initialize generator if API key exists
        if self.api_key:
            try:
                self.generator = OpenRouterMusicGenerator(self.api_key)
                self.update_status("API key loaded successfully")
            except Exception as e:
                self.update_status(f"Error initializing generator: {str(e)}")
        
        # Add message queue for thread communication
        self.message_queue = queue.Queue()
        self.root.after(100, self._process_messages)
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling."""
        self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def find_lmms(self):
        """Find LMMS installation path."""
        system = os.name
        if system in self.default_lmms_paths:
            for path in self.default_lmms_paths[system]:
                if os.path.exists(path):
                    if hasattr(self, 'status_text') and self.status_text:
                        self.update_status(f"Found LMMS at: {path}")
                    return path
        
        if hasattr(self, 'status_text') and self.status_text:
            self.update_status("LMMS not found in default locations. Please set the path manually.")
        return None
    
    def create_lmms_section(self):
        """Create the LMMS configuration section."""
        # Create a frame for LMMS settings
        lmms_frame = ttk.LabelFrame(self.main_frame, text="LMMS Configuration", padding=10)
        lmms_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # LMMS path
        ttk.Label(lmms_frame, text="LMMS Path:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.lmms_path_var = tk.StringVar(value=self.config.get_lmms_path())
        ttk.Entry(lmms_frame, textvariable=self.lmms_path_var, width=40).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(lmms_frame, text="Browse", command=self._browse_lmms).grid(row=0, column=2, padx=5, pady=2)
        
        # Output directory
        ttk.Label(lmms_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.output_dir_var = tk.StringVar(value=self.config.get_output_dir())
        ttk.Entry(lmms_frame, textvariable=self.output_dir_var, width=40).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(lmms_frame, text="Browse", command=self._browse_output_dir).grid(row=1, column=2, padx=5, pady=2)
        
        # Save settings button
        ttk.Button(lmms_frame, text="Save Settings", command=self._save_settings).grid(row=2, column=1, pady=10)
        
        # Configure grid weights
        lmms_frame.grid_columnconfigure(1, weight=1)
    
    def _browse_lmms(self):
        """Open file dialog to select LMMS executable."""
        if os.name == 'nt':  # Windows
            filetypes = [("Executable files", "*.exe"), ("All files", "*.*")]
        else:  # Linux/macOS
            filetypes = [("All files", "*")]
        
        path = filedialog.askopenfilename(
            title="Select LMMS Executable",
            filetypes=filetypes
        )
        
        if path:
            self.lmms_path_var.set(path)
            self.lmms_path = path
            self.config.set_lmms_path(path)
            self.update_status(f"LMMS path set to: {path}")
    
    def test_lmms(self):
        """Test if LMMS can be launched."""
        path = self.lmms_path_var.get().strip()
        if not path:
            messagebox.showwarning("Warning", "Please set LMMS path first!")
            return
        
        try:
            # Try to launch LMMS with --version flag
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                messagebox.showinfo("Success", "LMMS is working correctly!")
                self.lmms_path = path
                self.config.set_lmms_path(path)
            else:
                messagebox.showerror("Error", "LMMS test failed. Please check the path.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to test LMMS: {str(e)}")
    
    def create_api_section(self):
        """Create the API configuration section."""
        api_frame = ttk.LabelFrame(self.main_frame, text="API Configuration", padding=10)
        api_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        # API Key
        ttk.Label(api_frame, text="OpenRouter API Key:").grid(row=0, column=0, sticky="w", pady=2)
        self.api_key_var = tk.StringVar(value=self.api_key or "")
        self.api_key_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, width=50, show="*")
        self.api_key_entry.grid(row=0, column=1, padx=5)
        
        # Save API Key Button
        ttk.Button(api_frame, text="Save API Key", command=self.save_api_key).grid(row=0, column=2, padx=5)
        
        # Model Selection
        ttk.Label(api_frame, text="AI Model:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value=self.config.get_last_model() or "anthropic/claude-3-opus-20240229")
        self.model_combo = ttk.Combobox(api_frame, textvariable=self.model_var, width=47)
        self.model_combo['values'] = [
            "anthropic/claude-3-opus-20240229",
            "anthropic/claude-3-sonnet-20240229",
            "google/gemini-pro",
            "openai/gpt-4-turbo-preview"
        ]
        self.model_combo.grid(row=1, column=1, padx=5, pady=5)
    
    def create_generation_section(self):
        """Create the music generation section."""
        generation_frame = ttk.LabelFrame(self.main_frame, text="Generation Parameters", padding=10)
        generation_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        # Prompt input
        ttk.Label(generation_frame, text="Prompt:").grid(row=0, column=0, sticky="w", pady=2)
        self.prompt_text = tk.Text(generation_frame, height=3, width=50, wrap=tk.WORD)
        self.prompt_text.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
        
        # Add scrollbar to prompt text
        prompt_scrollbar = ttk.Scrollbar(generation_frame, orient="vertical", command=self.prompt_text.yview)
        prompt_scrollbar.grid(row=1, column=2, sticky="ns")
        self.prompt_text.configure(yscrollcommand=prompt_scrollbar.set)
        
        # Tempo input
        ttk.Label(generation_frame, text="Tempo (BPM):").grid(row=2, column=0, sticky="w", pady=2)
        self.tempo_var = tk.StringVar(value="120")
        tempo_entry = ttk.Entry(generation_frame, textvariable=self.tempo_var, width=10)
        tempo_entry.grid(row=2, column=1, sticky="w", pady=2)
        
        # Key selection
        ttk.Label(generation_frame, text="Key:").grid(row=3, column=0, sticky="w", pady=2)
        self.key_var = tk.StringVar(value="C")
        key_combo = ttk.Combobox(generation_frame, textvariable=self.key_var, 
                                values=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
                                width=10, state="readonly")
        key_combo.grid(row=3, column=1, sticky="w", pady=2)
        
        # Configure grid weights
        generation_frame.columnconfigure(0, weight=1)
        generation_frame.columnconfigure(1, weight=1)
    
    def create_controls_section(self):
        """Create the controls section."""
        controls_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding="5")
        controls_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Generate Button
        self.generate_button = ttk.Button(
            controls_frame,
            text="Generate Music",
            command=self.generate_music,
            style='Accent.TButton'
        )
        self.generate_button.grid(row=0, column=0, padx=5)
        
        # Export Button (initially disabled)
        self.export_button = ttk.Button(
            controls_frame,
            text="Export MIDI",
            command=self.export_midi,
            state=tk.DISABLED
        )
        self.export_button.grid(row=0, column=1, padx=5)
        
        # Open in LMMS Button (initially disabled)
        self.lmms_button = ttk.Button(
            controls_frame,
            text="Open in LMMS",
            command=self.open_in_lmms,
            state=tk.DISABLED
        )
        self.lmms_button.grid(row=0, column=2, padx=5)
    
    def create_status_section(self):
        """Create the status section."""
        status_frame = ttk.LabelFrame(self.main_frame, text="Status", padding="5")
        status_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Status Text
        self.status_text = tk.Text(status_frame, height=4, width=70, wrap=tk.WORD)
        self.status_text.grid(row=0, column=0, padx=5, pady=5)
        self.status_text.config(state=tk.DISABLED)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.status_text['yscrollcommand'] = scrollbar.set
    
    def set_default_values(self):
        self.current_midi = None
        self.update_status("Ready to generate music!")
    
    def save_api_key(self):
        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning("Warning", "Please enter an API key!")
            return
        
        try:
            # Test the API key by creating a generator
            self.generator = OpenRouterMusicGenerator(api_key=api_key)
            self.api_key = api_key
            self.config.set_api_key(api_key)
            self.update_status("API key saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid API key: {str(e)}")
    
    def _process_messages(self):
        """Process messages from the worker thread."""
        try:
            while True:
                message = self.message_queue.get_nowait()
                message_type, data = message
                
                if message_type == 'success':
                    midi_data, output_path = data
                    self.current_midi = midi_data
                    self.update_status(f"Music generated successfully!\nSaved to: {output_path}")
                    self.export_button.config(state=tk.NORMAL)
                    self.lmms_button.config(state=tk.NORMAL)
                elif message_type == 'error':
                    self.update_status(f"Error generating music: {str(data)}")
                    self.export_button.config(state=tk.DISABLED)
                    self.lmms_button.config(state=tk.DISABLED)
                
                self.message_queue.task_done()
        except queue.Empty:
            pass
        finally:
            self.generate_button.config(state=tk.NORMAL)
            self.root.after(100, self._process_messages)
    
    def _generate_music_worker(self, prompt, tempo, key, model):
        """Worker thread for music generation."""
        try:
            # Generate music
            midi_data = self.generator.generate_midi(
                prompt=prompt,
                tempo=tempo,
                key=key,
                model_id=model
            )
            
            # Save to output directory
            output_dir = self.config.get_output_dir()
            output_name = f"{prompt[:30].replace(' ', '_')}.mid"
            output_path = os.path.join(output_dir, output_name)
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Save MIDI file
            midi_data.write(output_path)
            
            # Return result through queue
            self.message_queue.put(("success", (midi_data, output_path)))
            
        except Exception as e:
            self.message_queue.put(("error", str(e)))
    
    def generate_music(self):
        """Generate music based on current settings."""
        # Get parameters
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Input Error", "Please enter a prompt for the music generation.")
            return
        
        try:
            tempo = int(self.tempo_var.get())
            if tempo < 40 or tempo > 240:
                raise ValueError("Tempo must be between 40 and 240 BPM")
        except ValueError as e:
            messagebox.showwarning("Input Error", str(e))
            return
        
        key = self.key_var.get()
        model = self.model_var.get()
        
        # Disable generate button during generation
        self.generate_button.config(state=tk.DISABLED)
        
        # Start generation in a separate thread
        thread = threading.Thread(
            target=self._generate_music_worker,
            args=(prompt, tempo, key, model)
        )
        thread.daemon = True
        thread.start()
    
    def export_midi(self):
        """Export the generated MIDI file."""
        if not self.current_midi:
            self.update_status("No music to export")
            return
        
        # Get output directory from config
        output_dir = self.config.get_output_dir()
        
        # Generate filename based on prompt
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        filename = f"{prompt[:30].replace(' ', '_')}.mid"
        filepath = os.path.join(output_dir, filename)
        
        try:
            self.current_midi.write(filepath)
            self.update_status(f"MIDI file exported to: {filepath}")
            messagebox.showinfo("Export Successful", f"MIDI file saved to:\n{filepath}")
        except Exception as e:
            self.update_status(f"Error exporting MIDI: {str(e)}")
            messagebox.showerror("Export Error", str(e))
    
    def open_in_lmms(self):
        """Open the generated MIDI file in LMMS."""
        if not self.current_midi:
            messagebox.showwarning("No Music", "Please generate music first.")
            return
        
        try:
            # Get LMMS path from config
            lmms_path = self.config.get_lmms_path()
            if not lmms_path:
                messagebox.showerror("LMMS Not Found", 
                    "LMMS path not configured. Please set it in the settings.")
                return
            
            # Validate LMMS path
            if not os.path.exists(lmms_path):
                messagebox.showerror("LMMS Not Found", 
                    f"LMMS not found at: {lmms_path}\nPlease update the path in settings.")
                return
            
            # Check if LMMS is executable
            if not os.access(lmms_path, os.X_OK):
                messagebox.showerror("Permission Error", 
                    f"Cannot execute LMMS at: {lmms_path}\nPlease check file permissions.")
                return
            
            # Save MIDI file to output directory
            output_dir = self.config.get_output_dir()
            output_name = "generated_music.mid"
            output_path = os.path.join(output_dir, output_name)
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Save MIDI file
            self.current_midi.write(output_path)
            
            # Launch LMMS with the MIDI file
            try:
                if sys.platform == "win32":
                    # On Windows, use the full path
                    subprocess.Popen([lmms_path, output_path])
                else:
                    # On Unix-like systems, use the executable name
                    subprocess.Popen([lmms_path, output_path])
                
                self.update_status(f"Opening {output_path} in LMMS...")
            except subprocess.SubprocessError as e:
                messagebox.showerror("Launch Error", 
                    f"Failed to launch LMMS: {str(e)}\nPlease check your LMMS installation.")
                return
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open LMMS: {str(e)}")
            return
    
    def update_status(self, message):
        """Update the status text."""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)

    def _browse_output_dir(self):
        """Open directory dialog to select output directory."""
        path = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_dir_var.get()
        )
        
        if path:
            self.output_dir_var.set(path)
            self.config.set_output_dir(path)
            self.update_status(f"Output directory set to: {path}")
    
    def _save_settings(self):
        """Save LMMS settings."""
        # Implementation of _save_settings method
        pass

def main():
    root = tk.Tk()
    app = SynthAIzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 