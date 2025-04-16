# -*- coding: utf-8 -*-
"""
Simple script to list available audio devices
"""

import sounddevice as sd

def main():
    """List all available audio devices"""
    print("=" * 80)
    print("Available Audio Devices:")
    print("=" * 80)
    
    devices = sd.query_devices()
    
    print("\nInput & Output Devices:")
    print("-" * 80)
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0 and device['max_output_channels'] > 0:
            print(f"ID {i}: {device['name']}")
            print(f"    Input channels: {device['max_input_channels']}")
            print(f"    Output channels: {device['max_output_channels']}")
            print(f"    Default samplerate: {device['default_samplerate']}")
            print()
    
    print("\nInput-only Devices:")
    print("-" * 80)
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0 and device['max_output_channels'] == 0:
            print(f"ID {i}: {device['name']}")
            print(f"    Input channels: {device['max_input_channels']}")
            print(f"    Default samplerate: {device['default_samplerate']}")
            print()
    
    print("\nOutput-only Devices:")
    print("-" * 80)
    for i, device in enumerate(devices):
        if device['max_input_channels'] == 0 and device['max_output_channels'] > 0:
            print(f"ID {i}: {device['name']}")
            print(f"    Output channels: {device['max_output_channels']}")
            print(f"    Default samplerate: {device['default_samplerate']}")
            print()
    
    # Check for default devices
    default_input = sd.query_devices(kind='input')
    default_output = sd.query_devices(kind='output')
    
    print("\nDefault Devices:")
    print("-" * 80)
    print(f"Default Input:  ID {default_input['index']} - {default_input['name']}")
    print(f"Default Output: ID {default_output['index']} - {default_output['name']}")

if __name__ == "__main__":
    main() 