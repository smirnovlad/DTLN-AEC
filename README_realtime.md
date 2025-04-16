# Real-time Audio Echo Cancellation (AEC)

This script enables real-time audio playback and processing with the DTLN-aec acoustic echo cancellation model.

## Features

- Real-time playback of reference audio through speakers
- Simultaneous microphone input capture and processing
- On-the-fly acoustic echo cancellation
- Sample rate conversion between input and output devices
- Saving of processed audio to file

## Requirements

1. Python 3.9 or higher
2. Install dependencies:
   ```
   python3.9 -m pip install -r requirements_realtime.txt
   ```
3. TFLite Runtime (for embedded devices) or TensorFlow:
   ```
   python3.9 -m pip install tflite-runtime
   # or
   python3.9 -m pip install tensorflow
   ```
4. Audio input device with at least 2 channels (mic + loopback)
5. DTLN-aec model files (see "Model Setup" section below)

## Model Setup

The script requires DTLN-aec model files to perform echo cancellation. These should be located in the DTLN-aec directory. The models come in three sizes:

- `dtln_aec_128` (1.8M parameters) - Fastest, good for low-powered devices
- `dtln_aec_256` (3.9M parameters) - Good balance of performance and quality
- `dtln_aec_512` (10.4M parameters) - Highest quality, requires more processing power

If the models are not already present, you can download them from the [DTLN-aec repository](https://github.com/breizhn/DTLN-aec).

Expected model location structure:
```
DTLN-AEC/
├── DTLN-aec/
│   ├── pretrained_models/
│   │   ├── dtln_aec_128_1.tflite
│   │   ├── dtln_aec_128_2.tflite
│   │   ├── dtln_aec_256_1.tflite
│   │   ├── dtln_aec_256_2.tflite
│   │   ├── dtln_aec_512_1.tflite
│   │   └── dtln_aec_512_2.tflite
...
```

## Usage

1. First, list available audio devices:
   ```
   python3.9 realtime_aec.py -l
   ```

2. Run the script with required parameters:
   ```
   python3.9 realtime_aec.py --input-device "DEVICE_NAME" --output-device "DEVICE_NAME" \
                            --reference-audio input.wav --result-path output.wav \
                            --duration 10 --use-aec=True
   ```

## Command-line Arguments

- `--input-device`, `-i` (required): Audio input device (ID or name substring)
- `--output-device`, `-o` (required): Audio output device (ID or name substring)
- `--reference-audio`, `-r` (required): Path to reference audio file to play
- `--result-path`, `-p` (required): Path to save processed output audio
- `--duration`, `-d`: Duration in seconds to process (default: entire file)
- `--model`, `-m`: Path to DTLN-aec model without extension (default: ./DTLN-aec/pretrained_models/dtln_aec_256)
- `--use-aec`: Enable/disable acoustic echo cancellation (default: True)
- `--latency`: Audio device latency in seconds (default: 0.2)
- `--threads`: Number of threads for TFLite interpreter (default: 1)
- `--input-channels`: Number of input channels (default: 2)
- `--measure`: Enable processing time measurement (default: False)

## Example Usage

### Listing audio devices:
```
python3.9 realtime_aec.py -l
```

### Basic usage with default parameters:
```
python3.9 realtime_aec.py -i "HDA Intel PCH" -o "default" \
                         -r music.wav -p processed.wav
```

### Process 30 seconds of audio, no AEC, with performance measurement:
```
python3.9 realtime_aec.py -i "HDA Intel PCH" -o "default" \
                         -r music.wav -p processed.wav \
                         -d 30 --use-aec=False --measure
```

### Using specific model with custom input channels:
```
python3.9 realtime_aec.py -i "Loopback" -o "default" \
                         -r music.wav -p processed.wav \
                         -m "./DTLN-aec/pretrained_models/dtln_aec_512" --input-channels 6
```

### Running without AEC (if models are not available):
```
python3.9 realtime_aec.py -i "HDA Intel PCH" -o "default" \
                         -r music.wav -p processed.wav \
                         --use-aec=False
```

## Device Configuration

Based on the available devices from `devices.txt`, good options include:

- For input: "HDA Intel PCH: CX8070 Analog (hw:0,0)" (ID 0)
- For loopback: "Loopback: PCM (hw:2,0)" (ID 4) or "Loopback: PCM (hw:2,1)" (ID 5)
- For output: "default" (ID 18) or "pulse" (ID 14)

If using a loopback device, ensure it has the reference signal on its last channel.

## Notes

1. For optimal results, the input device should have a loopback channel as its last channel
2. Processing time should be less than 8ms per block for real-time operation
3. The algorithm works at 16kHz, so input/output at different sample rates will be resampled
4. If you encounter issues with buffer underruns, try increasing the latency parameter
5. If models are not available and you still want to process audio, use `--use-aec=False` 