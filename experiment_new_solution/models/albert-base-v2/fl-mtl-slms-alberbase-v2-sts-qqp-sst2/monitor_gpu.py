#!/usr/bin/env python3
"""
GPU Monitoring Script for Federated Learning
Monitors GPU usage and memory across both GPUs during training
"""
import os
import time
import datetime
from typing import Dict, List
import subprocess

def get_gpu_info() -> List[Dict]:
    """Get GPU information using nvidia-smi"""
    try:
        # Run nvidia-smi command to get GPU info
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print(f"Error running nvidia-smi: {result.stderr}")
            return []
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:
                    gpu_info.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'memory_used': int(parts[2]),
                        'memory_total': int(parts[3]),
                        'utilization': int(parts[4]),
                        'temperature': int(parts[5]),
                        'power': float(parts[6])
                    })
        
        return gpu_info
    
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []

def format_gpu_info(gpu_info: List[Dict]) -> str:
    """Format GPU information for display"""
    if not gpu_info:
        return "No GPU information available"
    
    output = []
    output.append("=" * 80)
    output.append(f"GPU Monitor - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("=" * 80)
    
    for gpu in gpu_info:
        memory_pct = (gpu['memory_used'] / gpu['memory_total']) * 100
        memory_bar = '█' * int(memory_pct / 5) + '░' * (20 - int(memory_pct / 5))
        util_bar = '█' * int(gpu['utilization'] / 5) + '░' * (20 - int(gpu['utilization'] / 5))
        
        output.append(f"GPU {gpu['index']}: {gpu['name']}")
        output.append(f"  Memory: {memory_bar} {gpu['memory_used']:>4}/{gpu['memory_total']:>4} MB ({memory_pct:>5.1f}%)")
        output.append(f"  Utilization: {util_bar} {gpu['utilization']:>3}%")
        output.append(f"  Temperature: {gpu['temperature']:>2}°C  Power: {gpu['power']:>6.1f}W")
        output.append("")
    
    return '\n'.join(output)

def monitor_gpus(interval: int = 5, duration: int = 3600):
    """Monitor GPUs for specified duration"""
    print(f"Starting GPU monitoring (interval: {interval}s, duration: {duration}s)")
    print("Press Ctrl+C to stop monitoring")
    print()
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            gpu_info = get_gpu_info()
            if gpu_info:
                # Clear screen (Windows)
                os.system('cls')
                print(format_gpu_info(gpu_info))
                print(f"Monitoring... (Time remaining: {int(duration - (time.time() - start_time))}s)")
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    
    print("GPU monitoring completed")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor GPU usage during federated learning")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    parser.add_argument("--duration", type=int, default=3600, help="Monitoring duration in seconds")
    parser.add_argument("--test", action="store_true", help="Quick test (10 seconds)")
    
    args = parser.parse_args()
    
    if args.test:
        monitor_gpus(interval=2, duration=10)
    else:
        monitor_gpus(interval=args.interval, duration=args.duration)

if __name__ == "__main__":
    main()
