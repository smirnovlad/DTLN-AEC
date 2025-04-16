#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time audio processing script with DTLN-aec acoustic echo cancellation.
This script plays a reference audio file through speakers and simultaneously
captures input from a microphone, processes it with AEC, and saves the result.

Features:
- Real-time processing with DTLN-aec model
- Sample rate conversion for different input/output devices
- Playback of reference audio through speakers
- Optional AEC processing
- Saving processed audio to file
- Optional saving of raw microphone input and loopback signals

Example call:
    $python realtime_aec.py --input-device "HDA Intel PCH" --output-device "default" \
                           --reference-audio input.wav --result-path output.wav \
                           --duration 10 --use-aec=True --model ./DTLN-aec/pretrained_models/dtln_aec_256

Example with saving all signals:
    $python realtime_aec.py --input-device "pulse" --output-device "default" \
                           --reference-audio input.wav --result-path output.wav --save-all

Author: Based on work by Nils L. Westhausen and sanebow
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import time
import threading
import argparse
import queue
import resampy
import sys
import signal

# Try importing tflite_runtime first, fall back to TF if not available
try:
    import tflite_runtime.interpreter as tflite
    print("Using TFLite Runtime for inference")
except ImportError:
    try:
        import tensorflow.lite as tflite
        print("Using TensorFlow Lite for inference")
    except ImportError:
        print("Error: Neither TFLite Runtime nor TensorFlow is installed.")
        print("Please install one of them with:")
        print("  python3.9 -m pip install tflite-runtime")
        print("  or")
        print("  python3.9 -m pip install tensorflow")
        sys.exit(1)

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

# Set block length and block shift for DTLN-aec
block_len = 512
block_shift = 128

# Create buffers
in_buffer = np.zeros((block_len)).astype("float32")
in_buffer_lpb = np.zeros((block_len)).astype("float32")
out_buffer = np.zeros((block_len)).astype('float32')

# Create parser for command line arguments
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-l', '--list-devices', action='store_true',
                    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument("--input-device", "-i", type=int_or_str, required=True,
                    help="input device (numeric ID or substring)")
parser.add_argument("--output-device", "-o", type=int_or_str, required=True,
                    help="output device (numeric ID or substring)")
parser.add_argument("--reference-audio", "-r", type=str, required=True,
                    help="path to reference audio file to be played")
parser.add_argument("--result-path", "-p", type=str, required=True,
                    help="path to save processed output audio")
parser.add_argument("--mic-path", type=str, default=None,
                    help="path to save microphone input (default: <result_path_prefix>_mic.wav)")
parser.add_argument("--loopback-path", type=str, default=None,
                    help="path to save loopback signal (default: <result_path_prefix>_loopback.wav)")
parser.add_argument("--duration", "-d", type=float, default=None,
                    help="duration in seconds to process (None for entire file)")
parser.add_argument("--model", "-m", type=str, default="./DTLN-aec/pretrained_models/dtln_aec_512",
                    help="path to DTLN-aec model without extension")
parser.add_argument("--use-aec", type=bool, default=True,
                    help="whether to use acoustic echo cancellation")
parser.add_argument("--latency", type=float, default=0.05,
                    help="latency of sound device")
parser.add_argument("--threads", type=int, default=1,
                    help="number of threads for interpreter")
parser.add_argument("--input-channels", type=int, default=2,
                    help="number of input channels (default assumes mic on ch0, lpb on last channel)")
parser.add_argument("--measure", action='store_true',
                    help="measure and report processing time")
parser.add_argument("--save-all", action='store_true',
                    help="save all audio signals (processed, mic input, loopback)")
args = parser.parse_args(remaining)

# Always save all audio files, regardless of flags
save_additional = True

# Create default paths for mic and loopback files if not specified
if args.mic_path is None:
    result_prefix = os.path.splitext(args.result_path)[0]
    args.mic_path = f"{result_prefix}_mic.wav"
if args.loopback_path is None:
    result_prefix = os.path.splitext(args.result_path)[0]
    args.loopback_path = f"{result_prefix}_loopback.wav"

# Only print if explicitly requested or paths were specified
if args.save_all or args.mic_path is not None or args.loopback_path is not None:
    print(f"Will save microphone input to: {args.mic_path}")
    print(f"Will save loopback signal to: {args.loopback_path}")

# Initialize mic_audio and loopback_audio buffers
mic_audio = []
loopback_audio = []

# Check if input file exists
if not os.path.isfile(args.reference_audio):
    parser.error(f"Reference audio file not found: {args.reference_audio}")

# Load the reference audio
reference_audio, ref_sr = sf.read(args.reference_audio)
# If stereo, take only first channel
if len(reference_audio.shape) > 1:
    reference_audio = reference_audio[:, 0]

# Get device info
try:
    input_device_info = sd.query_devices(args.input_device, 'input')
    output_device_info = sd.query_devices(args.output_device, 'output')
    input_sr = int(input_device_info['default_samplerate'])
    output_sr = int(output_device_info['default_samplerate'])
    input_channels = min(args.input_channels, input_device_info['max_input_channels'])
    
    print(f"Input device: {input_device_info['name']} with {input_channels} channels @ {input_sr} Hz")
    print(f"Output device: {output_device_info['name']} @ {output_sr} Hz")
    print(f"Reference audio: {args.reference_audio} @ {ref_sr} Hz")
    
    # Check if we have enough input channels for mic and loopback
    if input_channels < 2:
        parser.error("Input device needs at least 2 channels (mic and loopback)")
except Exception as e:
    parser.error(f"Error getting device info: {e}")

# Init TFLite interpreters if using AEC
if args.use_aec:
    try:
        # Check if model files exist
        model_path_1 = f"{args.model}_1.tflite"
        model_path_2 = f"{args.model}_2.tflite"
        
        if not os.path.isfile(model_path_1) or not os.path.isfile(model_path_2):
            print(f"Error: Model files not found at {args.model}_1.tflite and/or {args.model}_2.tflite")
            print("Please check that the model files exist and the path is correct.")
            print("Common model locations are:")
            print("  - ./DTLN-aec/pretrained_models/dtln_aec_128")
            print("  - ./DTLN-aec/pretrained_models/dtln_aec_256")
            print("  - ./DTLN-aec/pretrained_models/dtln_aec_512")
            
            if args.use_aec:
                print("\nSince AEC is enabled but models are not found, exiting.")
                sys.exit(1)
        
        print(f"Loading models from {model_path_1} and {model_path_2}")
        
        interpreter_1 = tflite.Interpreter(
            model_path=model_path_1, num_threads=args.threads)
        interpreter_1.allocate_tensors()
        interpreter_2 = tflite.Interpreter(
            model_path=model_path_2, num_threads=args.threads)
        interpreter_2.allocate_tensors()
        
        # Get input/output details
        input_details_1 = interpreter_1.get_input_details()
        output_details_1 = interpreter_1.get_output_details()
        input_details_2 = interpreter_2.get_input_details()
        output_details_2 = interpreter_2.get_output_details()
        
        # Initialize LSTM states
        states_1 = np.zeros(input_details_1[1]["shape"]).astype("float32")
        states_2 = np.zeros(input_details_2[1]["shape"]).astype("float32")
        
        print("AEC model loaded successfully")
    except Exception as e:
        print(f"Error loading AEC model: {e}")
        print("\nIf you don't have the models, you can either:")
        print("1. Download them from the DTLN-aec repository")
        print("2. Run with --use-aec=False to disable AEC")
        sys.exit(1)

# Resample reference audio to output device sample rate if necessary
if ref_sr != output_sr:
    print(f"Resampling reference audio from {ref_sr} Hz to {output_sr} Hz")
    reference_audio = resampy.resample(reference_audio, ref_sr, output_sr)

# Create queues for inter-thread communication
playback_queue = queue.Queue()
recording_queue = queue.Queue()
result_queue = queue.Queue()
playback_finished = threading.Event()
recording_finished = threading.Event()
processing_finished = threading.Event()

# Flag to indicate that we're shutting down due to Ctrl+C
interrupted = False

# Limit duration if specified
if args.duration is not None:
    max_samples = int(args.duration * output_sr)
    if len(reference_audio) > max_samples:
        reference_audio = reference_audio[:max_samples]
    print(f"Limiting playback to {args.duration} seconds")

# Create buffers to store results
processed_audio = []
processing_times = []

# Custom Ctrl+C handler
def signal_handler(sig, frame):
    global interrupted
    print("\nInterrupted by user, saving immediately...")
    interrupted = True
    
    # Set these events to signal all threads to finish
    recording_finished.set()
    playback_finished.set()
    processing_finished.set()  # Force processing to stop immediately too
    
    # Immediately stop the stream if it's running
    if 'stream' in globals() and stream is not None and stream.active:
        stream.stop()
    
    # Let the main thread handle the cleanup and saving
    # Actual exit happens in the finally block

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Playback thread function
def playback_thread():
    # Calculate how many samples to send at a time
    # We'll match the block_shift of the processing (128 samples at 16kHz)
    samples_per_block = int(block_shift * output_sr / 16000)
    
    try:
        # Send audio in chunks to the playback queue
        for i in range(0, len(reference_audio), samples_per_block):
            if interrupted:
                break
            block = reference_audio[i:i+samples_per_block]
            # Pad last block if necessary
            if len(block) < samples_per_block:
                block = np.pad(block, (0, samples_per_block - len(block)))
            playback_queue.put(block)
            
        # Signal that playback is complete
        playback_finished.set()
    except Exception as e:
        print(f"Error in playback thread: {e}")
        playback_finished.set()

# Processing thread function
def processing_thread():
    global in_buffer, in_buffer_lpb, out_buffer, states_1, states_2, processed_audio
    global mic_audio
    
    # DTLN requires 16kHz sample rate
    target_sr = 16000
    
    try:
        while not (recording_finished.is_set() and recording_queue.empty()):
            # Get next block from recording queue
            try:
                data = recording_queue.get(timeout=1.0)
                mic_signal = data['mic']
                lpb_signal = data['lpb']
                
                # Always store raw mic signal before resampling
                mic_audio.append(mic_signal.copy())
                
                # Resample to 16kHz if needed
                if input_sr != target_sr:
                    mic_signal = resampy.resample(mic_signal, input_sr, target_sr)
                    lpb_signal = resampy.resample(lpb_signal, input_sr, target_sr)
                
                # Check and fix signal lengths if needed
                if len(mic_signal) != block_shift:
                    # Print warning only the first time
                    if not hasattr(processing_thread, 'warned_size'):
                        print(f"Warning: Resampled block size {len(mic_signal)} doesn't match expected {block_shift}. Adjusting...")
                        processing_thread.warned_size = True
                    
                    # Fix the signal length by padding or trimming
                    if len(mic_signal) < block_shift:
                        # Pad with zeros
                        mic_signal = np.pad(mic_signal, (0, block_shift - len(mic_signal)))
                    else:
                        # Trim
                        mic_signal = mic_signal[:block_shift]
                
                if len(lpb_signal) != block_shift:
                    # Fix the signal length by padding or trimming
                    if len(lpb_signal) < block_shift:
                        # Pad with zeros
                        lpb_signal = np.pad(lpb_signal, (0, block_shift - len(lpb_signal)))
                    else:
                        # Trim
                        lpb_signal = lpb_signal[:block_shift]
                
                if args.measure:
                    start_time = time.time()
                
                # Process with AEC if enabled
                if args.use_aec:
                    # Update buffers
                    in_buffer[:-block_shift] = in_buffer[block_shift:]
                    in_buffer[-block_shift:] = mic_signal
                    
                    in_buffer_lpb[:-block_shift] = in_buffer_lpb[block_shift:]
                    in_buffer_lpb[-block_shift:] = lpb_signal
                    
                    # Calculate FFT of input block
                    in_block_fft = np.fft.rfft(np.squeeze(in_buffer)).astype("complex64")
                    # Create magnitude
                    in_mag = np.abs(in_block_fft)
                    in_mag = np.reshape(in_mag, (1, 1, -1)).astype("float32")
                    
                    # Calculate log pow of loopback
                    lpb_block_fft = np.fft.rfft(np.squeeze(in_buffer_lpb)).astype("complex64")
                    lpb_mag = np.abs(lpb_block_fft)
                    lpb_mag = np.reshape(lpb_mag, (1, 1, -1)).astype("float32")
                    
                    # Set tensors to the first model
                    interpreter_1.set_tensor(input_details_1[0]["index"], in_mag)
                    interpreter_1.set_tensor(input_details_1[2]["index"], lpb_mag)
                    interpreter_1.set_tensor(input_details_1[1]["index"], states_1)
                    
                    # Run calculation
                    interpreter_1.invoke()
                    
                    # Get the output of the first block
                    out_mask = interpreter_1.get_tensor(output_details_1[0]["index"])
                    states_1 = interpreter_1.get_tensor(output_details_1[1]["index"])
                    
                    # Apply mask and calculate the IFFT
                    estimated_block = np.fft.irfft(in_block_fft * out_mask)
                    
                    # Reshape the time domain frames
                    estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype("float32")
                    in_lpb = np.reshape(in_buffer_lpb, (1, 1, -1)).astype("float32")
                    
                    # Set tensors to the second block
                    interpreter_2.set_tensor(input_details_2[1]["index"], states_2)
                    interpreter_2.set_tensor(input_details_2[0]["index"], estimated_block)
                    interpreter_2.set_tensor(input_details_2[2]["index"], in_lpb)
                    
                    # Run calculation
                    interpreter_2.invoke()
                    
                    # Get output tensors
                    out_block = interpreter_2.get_tensor(output_details_2[0]["index"])
                    states_2 = interpreter_2.get_tensor(output_details_2[1]["index"])
                    
                    # Shift values and update buffer
                    out_buffer[:-block_shift] = out_buffer[block_shift:]
                    out_buffer[-block_shift:] = np.zeros((block_shift))
                    out_buffer += np.squeeze(out_block)
                    
                    # Save block_shift samples to result
                    processed_block = out_buffer[:block_shift].copy()
                else:
                    # If AEC is disabled, just pass through the mic signal
                    processed_block = mic_signal
                
                # Append to results
                processed_audio.append(processed_block)
                
                if args.measure:
                    processing_times.append(time.time() - start_time)
                    
                # Put processed data in result queue
                result_queue.put(processed_block)
                
            except queue.Empty:
                continue
                
        # Signal that processing is complete
        processing_finished.set()
    except Exception as e:
        print(f"Error in processing thread: {e}")
        processing_finished.set()

# Audio callback function
def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(f"Status: {status}")
    
    # Get audio block for playback from queue if available
    try:
        playback_data = playback_queue.get_nowait()
        # Ensure correct shape for output
        outdata[:, 0] = playback_data.reshape(-1)
        
        # Store playback data (reference audio) as loopback - always store at original sample rate
        if save_additional:
            # Always store the exact playback data at output_sr
            loopback_audio.append(playback_data.copy())
                
    except queue.Empty:
        # If no more playback data, output zeros and set finished
        outdata.fill(0)
        if playback_finished.is_set():
            recording_finished.set()
            raise sd.CallbackStop
    
    # Extract mic signal from input
    mic_signal = indata[:, 0]  # First channel is microphone
    
    # Put data in recording queue for processing
    # We no longer use the last channel as loopback, since we're collecting playback directly
    recording_queue.put({
        'mic': mic_signal.copy(),
        'lpb': playback_data.copy() if 'playback_data' in locals() else np.zeros_like(mic_signal)
    })

# Function to save processed audio
def save_processed_audio():
    global processed_audio, mic_audio, loopback_audio
    
    results = []
    
    # Save processed audio
    if processed_audio:
        try:
            # Concatenate all processed blocks
            processed_result = np.concatenate(processed_audio)
            
            # Check for clipping
            if np.max(np.abs(processed_result)) > 1.0:
                scaling_factor = 0.99 / np.max(np.abs(processed_result))
                processed_result = processed_result * scaling_factor
                print(f"Output was clipping, scaled by factor {scaling_factor:.4f}")
            
            # Save to file
            sf.write(args.result_path, processed_result, 16000)
            print(f"Processed audio saved to {args.result_path}")
            results.append(f"Processed: {args.result_path}")
        except Exception as e:
            print(f"Error saving processed audio: {e}")
    else:
        print("No processed audio to save")
    
    # Always save microphone input
    if mic_audio:
        try:
            # Concatenate all mic blocks
            mic_result = np.concatenate(mic_audio)
            
            # Save to file
            sf.write(args.mic_path, mic_result, input_sr)
            print(f"Microphone input saved to {args.mic_path}")
            results.append(f"Microphone: {args.mic_path}")
        except Exception as e:
            print(f"Error saving microphone input: {e}")
    else:
        print("No microphone audio to save")
    
    # Always save loopback signal - keep the entire signal!
    if loopback_audio:
        try:
            # Concatenate all loopback blocks
            loopback_result = np.concatenate(loopback_audio)
            
            # Don't trim the loopback to match mic_result anymore
            # This was causing the loopback to be shorter
            
            # Save to file using the output_sr (the original sample rate of the audio)
            sf.write(args.loopback_path, loopback_result, output_sr)
            print(f"Loopback signal saved to {args.loopback_path} ({len(loopback_result)/output_sr:.2f}s)")
            results.append(f"Loopback: {args.loopback_path}")
        except Exception as e:
            print(f"Error saving loopback signal: {e}")
    else:
        print("No loopback audio to save")
    
    # Print summary of saved files
    if results:
        print("\nSaved audio files:")
        for result in results:
            print(f"- {result}")
        return True
    else:
        print("\nNo audio files were saved!")
        return False

# Function to process remaining data in queue
def process_remaining_data():
    global processed_audio, mic_audio
    
    if not recording_queue.empty():
        print(f"Processing remaining {recording_queue.qsize()} blocks from recording queue...")
        
        # Process until queue is empty
        while not recording_queue.empty():
            try:
                # Get data from queue without waiting (we know there's data)
                data = recording_queue.get_nowait()
                
                # Store mic signals
                mic_audio.append(data['mic'].copy())
                
                # For processed audio, we just store the mic signal directly
                # This is a simplified approach since we can't run the full AEC in this emergency mode
                if input_sr != 16000:
                    processed_signal = resampy.resample(data['mic'], input_sr, 16000)
                else:
                    processed_signal = data['mic']
                
                # Adjust length if needed
                if len(processed_signal) != block_shift:
                    if len(processed_signal) < block_shift:
                        processed_signal = np.pad(processed_signal, (0, block_shift - len(processed_signal)))
                    else:
                        processed_signal = processed_signal[:block_shift]
                
                processed_audio.append(processed_signal)
                
            except Exception as e:
                print(f"Error processing remaining data: {e}")
                break

# Start processing thread
proc_thread = threading.Thread(target=processing_thread)
proc_thread.start()

# Start playback thread
play_thread = threading.Thread(target=playback_thread)
play_thread.start()

# Open audio stream
stream = None
try:
    stream = sd.Stream(device=(args.input_device, args.output_device),
                  samplerate=input_sr,  # Use input sample rate for the stream
                  blocksize=int(block_shift * input_sr / 16000),  # Scale blocksize based on sample rate
                  dtype=np.float32,
                  latency=args.latency,
                  channels=(input_channels, 1),
                  callback=audio_callback)
    
    with stream:
        print('#' * 80)
        print('Stream started - Press Ctrl+C to stop and save')
        print('#' * 80)
        
        # Wait for all processing to complete
        recording_finished.wait()
        processing_finished.wait()
        
except Exception as e:
    print(f"Error in audio stream: {e}")
finally:
    print("\nCleaning up and saving data...")
    
    # Make sure to stop the stream if it's still running
    if stream is not None and stream.active:
        print("Stopping audio stream...")
        stream.stop()
    
    # Force all finished flags to be set (in case of error)
    recording_finished.set()
    playback_finished.set()
    
    if interrupted:
        # When interrupted, don't wait for processing thread - save immediately
        processing_finished.set()
        print("Saving collected data immediately...")
    else:
        # Only wait for processing thread if not interrupted
        if proc_thread.is_alive():
            print("Waiting for processing to complete (timeout: 5.0s)...")
            processing_finished.wait(timeout=5.0)
            
            # If still not finished, we still want to save what we have
            if not processing_finished.is_set():
                print("Processing thread did not finish in time, saving collected data anyway...")
    
    # Skip processing remaining data if interrupted to exit faster
    if interrupted and False:  # Disabled for faster exit
        process_remaining_data()
    
    # Save all audio data
    print("Saving audio files...")
    save_processed_audio()
    
    if interrupted:
        print("Exit due to user interrupt.")
        
    # Print performance statistics if measuring
    if args.measure and processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)
        print(f"Processing performance:")
        print(f"  Average time: {avg_time * 1000:.2f} ms per block")
        print(f"  Maximum time: {max_time * 1000:.2f} ms per block")
        print(f"  Total blocks processed: {len(processing_times)}")
        
        # Warn if processing is too slow
        if avg_time > (block_shift / 16000):
            print(f"WARNING: Average processing time ({avg_time * 1000:.2f} ms) is greater than")
            print(f"         the block time ({block_shift / 16000 * 1000:.2f} ms at 16kHz).")
            print(f"         Real-time processing is not possible with current settings.")

print("Done!") 