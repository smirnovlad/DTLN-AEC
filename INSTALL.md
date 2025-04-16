# Installation and Setup Instructions

This document provides instructions for setting up the environment and installing the necessary packages for the real-time audio processing script with AEC.

## System Requirements

- Python 3.9 or later
- Linux (tested on Ubuntu)
- Audio devices for input (microphone) and output (speakers)

## Installation Steps

1. Clone the repository (if applicable):
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install Python 3.9 if not already installed:
   ```bash
   # For Ubuntu/Debian
   sudo apt update
   sudo apt install python3.9 python3.9-venv python3.9-dev
   ```

3. Create a virtual environment (recommended):
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate
   ```

4. Install required packages:
   ```bash
   python3.9 -m pip install -r requirements.txt
   ```

   Alternatively, install packages manually:
   ```bash
   python3.9 -m pip install sounddevice soundfile numpy scipy
   python3.9 -m pip install tflite-runtime
   ```

## Verifying Installation

1. Check available audio devices:
   ```bash
   python3.9 rt_audio_aec.py -l
   ```

2. Test AEC model loading:
   ```bash
   python3.9 test_aec_model.py -a
   ```

## Troubleshooting

### Common Issues

1. **Cannot find sound devices**:
   - Ensure your audio devices are properly connected and recognized by the system
   - Check with `aplay -l` and `arecord -l` to see if Linux recognizes the devices
   - Make sure your user has permission to access audio devices

2. **TFLite runtime installation issues**:
   - If installing tflite-runtime fails, you can try installing TensorFlow instead:
     ```bash
     python3.9 -m pip install tensorflow
     ```
   - Then modify the script imports accordingly

3. **Performance issues**:
   - If you experience buffer underflow/overflow:
     - Try a smaller model (128 or 256 instead of 512)
     - Increase the latency parameter
     - Use a more powerful CPU
     - Close other CPU-intensive applications

### Advanced Setup for Loopback

If your hardware doesn't provide a loopback channel, you can set up a virtual loopback device:

1. Load the snd-aloop kernel module:
   ```bash
   sudo modprobe snd_aloop
   ```

2. Add the module to load on boot:
   ```bash
   echo "snd-aloop" | sudo tee -a /etc/modules
   ```

3. Check if the loopback device is created:
   ```bash
   arecord -l | grep Loopback
   ```

4. Use the loopback device for testing. 