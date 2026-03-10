"""
Project: High-Dimensional Polynomial Randomness Framework (HDPRF)
File: src/main.py
Type: Open Source Release
Version: 1.5.5 (GitHub Ready / English Localization)
Description:
    Core execution script for the HDPRF.
    Refactored for portability, CLI execution, and semantic variable naming.
    
    Change Log:
    1. Algorithm Logic Fix: Introduced global_offset to prevent systemic replay.
    2. JIT Optimization: Eliminated dynamic array allocation inside kernels.
    3. Path Portability: Adaptive path resolution for standard repo layouts.
    4. Entropy Feature: Integrated OS-level fallback and external entropy injection.
    5. Localization: Fully translated comments and logs to English.
"""

import io
import os
import sys
import json
import time
import numba
import random
import pstats
import hashlib
import cProfile
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any, Generator

import numpy as np 
from numba import jit, prange, int8, int64, float64, uint8, uint64

# ==============================================================================
# Control Logic Layer
# ==============================================================================

def read_config(file_path: Path) -> Dict[str, Any]:
    """Read and validate the JSON configuration file."""
    try:
        with open(file_path, "r", encoding='utf-8') as file:
            config = json.load(file)
            
        required_keys = ["F1", "F_layers", "dimension"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in configuration: '{key}'")
                
        config["iteration_count"] = config.get("iteration_count", 1)
        config["enable_dim_check"] = config.get("enable_dim_check", 0)
        config["dim_verification_map"] = config.get("dim_verification_map", [])
            
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in configuration file: {e}")

def get_external_seed(file_path: Path) -> int:
    """Read external binary file and convert to integer seed via SHA-512."""
    if not file_path.exists():
        raise FileNotFoundError(f"External entropy source not found: {file_path}")
    
    with open(file_path, "rb") as f:
        entropy_data = f.read()
        
    # Condense arbitrary file size into a 64-byte secure digest
    digest = hashlib.sha512(entropy_data).digest()
    return int.from_bytes(digest, byteorder='big')

def resolve_polynomial_params(config: Dict[str, Any], target_layer_index: int) -> Tuple[Dict[str, Any], str]:
    """
    Resolve parameters for the target layer. 
    If undefined, fallback to the nearest available lower layer setting (Adaptive Fallback).
    """
    f_layers = config.get("F_layers", {})
    current_search_index = target_layer_index
    
    while current_search_index >= 2:
        layer_key = f"F{current_search_index}"
        if layer_key in f_layers:
            return f_layers[layer_key], layer_key
        current_search_index -= 1
        
    raise ValueError(f"Unable to resolve polynomial parameters for F{target_layer_index}, and fallback failed.")

def verify_dimension(expected_dim_map: List[Any], layer_index: int, actual_dim: int) -> None:
    """Verify dimension count based on dim_verification_map."""
    map_index = layer_index - 1 
    if map_index < len(expected_dim_map):
        expected_val = expected_dim_map[map_index]
        if expected_val not in [None, "-", "null", ""]:
            try:
                expected_int = int(expected_val)
                if actual_dim != expected_int:
                    raise ValueError(f"Dimension verification failed (F{layer_index}): Expected {expected_int}, Got {actual_dim}")
            except ValueError as e:
                raise ValueError(str(e))

def run_intermediate_layer(state_matrix: List[np.ndarray], dimension: int, f_config: Dict[str, Any], block_length: int) -> List[np.ndarray]:
    """Intermediate layer wrapper: Generates the next-generation state matrix for double buffering."""
    f_coeffs_np = np.array(f_config["coeffs"], dtype=np.int64)
    f_powers_np = np.array(f_config["powers"], dtype=np.int64)
    f_constant = np.int64(f_config["constant"])

    fixed_axis_source = state_matrix[:dimension]
    fixed_seed_source = state_matrix[dimension:dimension*2]
    
    max_len_axis = max(len(s) for s in fixed_axis_source) if fixed_axis_source else 0
    max_len_seed = max(len(s) for s in fixed_seed_source) if fixed_seed_source else 0
    
    axis_array = np.zeros((len(fixed_axis_source), max_len_axis), dtype=np.uint8)
    for i, s in enumerate(fixed_axis_source):
        if len(s) > 0: axis_array[i, :len(s)] = s
            
    seed_array = np.zeros((len(fixed_seed_source), max_len_seed), dtype=np.uint8)
    for i, s in enumerate(fixed_seed_source):
        if len(s) > 0: seed_array[i, :len(s)] = s

    axis_lengths = np.array([len(s) for s in fixed_axis_source], dtype=np.int64)
    seed_lengths = np.array([len(s) for s in fixed_seed_source], dtype=np.int64)

    total_bytes_needed = 2 * dimension * block_length

    # Pass global_offset = 0 since intermediate layers generate a fixed-length block
    s_output_1d = numba_generate_digits_core(
        total_bytes_needed, dimension, axis_array, axis_lengths, seed_array, seed_lengths, 
        f_coeffs_np, f_powers_np, f_constant, np.uint64(0)
    )
    
    s_output_2d = s_output_1d.reshape((2 * dimension, block_length))
    return [row for row in s_output_2d]

def read_primes(file_path: Path) -> List[int]:
    try:
        with open(file_path, "r", encoding='utf-8') as file:
            primes = [int(line.strip()) for line in file if line.strip().isdigit()]
        if not primes: 
            raise ValueError("Prime file is empty")
        return primes
    except FileNotFoundError:
        raise FileNotFoundError(f"Prime file not found: {file_path}. Please ensure it exists in the 'data/' directory.")
    except Exception as e: 
        raise Exception(f"Error reading prime file: {e}")
    
def get_extractor(name: str):
    if name == "sha256": 
        return hashlib.sha256
    elif name == "sha512": 
        return hashlib.sha512
    elif name == "blake2b": 
        return hashlib.blake2b
    elif name in ["none", None]: 
        return None
    else: raise ValueError(f"Unsupported Extractor: {name}")

def generate_prime_combinations(primes: List[int], num_variables: int, total_groups: int) -> List[List[int]]:
    if num_variables > len(primes):
        num_variables = len(primes)
    if num_variables <= 0: 
        return [[]] * total_groups
    
    return [random.sample(primes, num_variables) for _ in range(total_groups)]

def compute_polynomial_values(prime_combinations: List[List[int]], f1_config: Dict[str, Any], extractor_name: str) -> List[int]:
    coeffs = f1_config["coeffs"]
    powers = f1_config["powers"]
    constant = f1_config["constant"]
    polynomial_values_raw = []
    
    for combination in prime_combinations:
        if len(combination) != len(coeffs):
            polynomial_values_raw.append(0.0)
            continue
        try:
            val = constant
            for c, p, v in zip(coeffs, powers, combination):
                val += c * (v ** p)
            polynomial_values_raw.append(float(val))
        except Exception:
            polynomial_values_raw.append(0.0)

    extractor = get_extractor(extractor_name)
    if not extractor: 
        return polynomial_values_raw

    extracted_values = []
    for val in polynomial_values_raw:
        digest = extractor(repr(val).encode('utf-8')).digest()
        if len(digest) < 8: digest = digest.ljust(8, b'\x00')
        extracted_values.append(int.from_bytes(digest[:8], 'big'))
        
    return extracted_values

def generate_initial_state_matrix(initial_entropy_seeds: List[int], length: int) -> List[np.ndarray]:
    input_array = np.array(initial_entropy_seeds, dtype=np.uint64)
    state_array = numba_core_uint8(input_array, length)
    
    return [row for row in state_array]

# ==============================================================================
# Low-Level Calculation Kernels (Numba)
# ==============================================================================

@jit(uint8[:,:](uint64[:], int64), nopython=True, parallel=True, cache=True)
def numba_core_uint8(input_array, length):
    num_values = input_array.shape[0]
    output_array = np.zeros((num_values, length), dtype=np.uint8)
    
    for i in prange(num_values):
        current_val = input_array[i] 
        for j in range(length):
            current_val = (current_val + np.uint64(0x9e3779b97f4a7c15))
            z = current_val
            z = (z ^ (z >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
            z = (z ^ (z >> 27)) * np.uint64(0x94d049bb133111eb)
            current_val = z ^ (z >> 31)
            output_array[i, j] = np.uint8(current_val & 0xFF)
            
    return output_array

@jit(uint8[:](int64, int64, uint8[:,:], int64[:], uint8[:,:], int64[:], int64[:], int64[:], int64, uint64), nopython=True, parallel=True, cache=True)
def numba_generate_digits_core(iterations_needed, D, fixed_axis_source_array, axis_lengths, fixed_seed_source_array, seed_lengths, f2_coeffs, f2_powers, f2_constant, global_offset):
    output_values = np.zeros(iterations_needed, dtype=np.uint8)
    num_coeffs = len(f2_coeffs)
    
    for i in prange(iterations_needed):
        # Apply global_offset to ensure continuous hypercube traversal across chunks
        current_global_index = np.uint64(i) + global_offset
        block_jump_sequence = current_global_index // 256
        step_in_block = current_global_index % 256
        
        f2_result = f2_constant
        
        # Axis Calculation: Scalar computation avoids dynamic allocation in threads
        for d_idx in range(D):
            s_total_len = axis_lengths[d_idx]
            if s_total_len == 0: continue
            num_blocks = np.uint64(s_total_len // 256)
            
            if num_blocks == 0:
                val = fixed_axis_source_array[d_idx, current_global_index % np.uint64(s_total_len)]
            else:
                rng_state = np.uint64(d_idx * 999999937) + block_jump_sequence
                z = rng_state + np.uint64(0x9e3779b97f4a7c15)
                z = (z ^ (z >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
                z = (z ^ (z >> 27)) * np.uint64(0x94d049bb133111eb)
                rng_val = z ^ (z >> 31)
                target_block_id = rng_val % num_blocks
                final_idx = target_block_id * 256 + step_in_block
                val = fixed_axis_source_array[d_idx, final_idx]

            # Polynomial expansion
            # Note: Exploits C-standard / NumPy 64-bit integer overflow semantics for mixing
            c_idx = d_idx % num_coeffs
            coeff = f2_coeffs[c_idx]
            power = f2_powers[c_idx]
            term = val ** power if power > 0 else 1
            f2_result += coeff * term
            
        f2_byte = f2_result & 0xFF 

        # Seed Source mixing
        num_seed_sources = fixed_seed_source_array.shape[0]
        if num_seed_sources > 0:
            seed_source_idx = (current_global_index % np.uint64(D)) % np.uint64(num_seed_sources)
            s_seed_len = seed_lengths[seed_source_idx]
            num_blocks_seed = np.uint64(s_seed_len // 256)
            
            if num_blocks_seed > 0:
                rng_state_seed = np.uint64(seed_source_idx * 1000000007) + block_jump_sequence
                z_s = rng_state_seed + np.uint64(0x9e3779b97f4a7c15)
                z_s = (z_s ^ (z_s >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
                z_s = (z_s ^ (z_s >> 27)) * np.uint64(0x94d049bb133111eb)
                rng_val_seed = z_s ^ (z_s >> 31)
                
                target_seed_block = rng_val_seed % num_blocks_seed
                seed_final_idx = target_seed_block * 256 + step_in_block
                seed_val = fixed_seed_source_array[seed_source_idx, seed_final_idx]
                output_values[i] = (seed_val + f2_byte) & 0xFF
            else:
                if s_seed_len > 0:
                    output_values[i] = (fixed_seed_source_array[seed_source_idx, current_global_index % np.uint64(s_seed_len)] + f2_byte) & 0xFF
                else:
                    output_values[i] = f2_byte
        else:
            output_values[i] = f2_byte
            
    return output_values

@jit(uint8[:](uint8[:], uint64, int64, int64), nopython=True, parallel=True, cache=True)
def numba_generate_bytes_from_digits(s1_values_np, state_counter, buffer_size, num_threads):
    num_bytes = len(s1_values_np) 
    output_bytes = np.zeros(num_bytes, dtype=np.uint8)
    
    LOGICAL_CHUNK_SIZE = 65536
    num_logical_chunks = (num_bytes + LOGICAL_CHUNK_SIZE - 1) // LOGICAL_CHUNK_SIZE
    
    for chunk_idx in prange(num_logical_chunks):
        start = chunk_idx * LOGICAL_CHUNK_SIZE
        end = min((chunk_idx + 1) * LOGICAL_CHUNK_SIZE, num_bytes)
        
        internal_state = np.uint64(state_counter) ^ np.uint64(chunk_idx + 1)
        
        STATE_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
        MIX_CONST_A = np.uint64(1103515245)
        MIX_CONST_B = np.uint64(15287)
        state_buffer = np.zeros(buffer_size, dtype=np.uint64)
        
        for k in range(buffer_size):
            internal_state = (internal_state * MIX_CONST_A + MIX_CONST_B) & STATE_MASK
            state_buffer[k] = internal_state
            
        for byte_idx in range(start, end):
            val_from_core = s1_values_np[byte_idx]
            buffer_write_idx = byte_idx % buffer_size
            internal_state = (internal_state * MIX_CONST_A + np.uint64(val_from_core) + MIX_CONST_B) & STATE_MASK
            state_buffer[buffer_write_idx] = internal_state
            mixed_scalar = state_buffer[0]
            
            for k in range(1, buffer_size):
                mixed_scalar = mixed_scalar ^ state_buffer[k]
                
            x = mixed_scalar
            x = x ^ (x >> 33); x = (x * np.uint64(0xff51afd7ed558ccd)) & STATE_MASK
            x = x ^ (x >> 33); x = (x * np.uint64(0xc4ceb9fe1a85ec53)) & STATE_MASK
            x = x ^ (x >> 33)
            
            output_byte_extracted = (x >> 56) & 0xFF
            output_bytes[byte_idx] = np.uint8(output_byte_extracted ^ val_from_core)
            
    return output_bytes

def generate_byte_stream(state_matrix: List[np.ndarray], dimension: int, f_config: Dict[str, Any], num_seeds: int, target_length: int, buffer_size: int) -> Generator[bytes, None, None]:
    f_coeffs_np = np.array(f_config["coeffs"], dtype=np.int64)
    f_powers_np = np.array(f_config["powers"], dtype=np.int64)
    f_constant = np.int64(f_config["constant"])

    fixed_axis_source = state_matrix[:dimension]
    fixed_seed_source = state_matrix[dimension:dimension*2]
    
    max_len_axis = max(len(s) for s in fixed_axis_source) if fixed_axis_source else 0
    max_len_seed = max(len(s) for s in fixed_seed_source) if fixed_seed_source else 0
    
    axis_array = np.zeros((len(fixed_axis_source), max_len_axis), dtype=np.uint8)
    for i, s in enumerate(fixed_axis_source):
        if len(s) > 0: axis_array[i, :len(s)] = s
    seed_array = np.zeros((len(fixed_seed_source), max_len_seed), dtype=np.uint8)
    for i, s in enumerate(fixed_seed_source):
        if len(s) > 0: seed_array[i, :len(s)] = s

    axis_lengths = np.array([len(s) for s in fixed_axis_source], dtype=np.int64)
    seed_lengths = np.array([len(s) for s in fixed_seed_source], dtype=np.int64)

    total_bytes_generated = 0   
    chunk_counter = 0
    BYTES_PER_CHUNK = 4 * 1024 * 1024 
    total_target_length = target_length * num_seeds
    num_threads = numba.get_num_threads()
    
    while total_bytes_generated < total_target_length:
        # Pass total_bytes_generated as global_offset to maintain state progression
        s_values_chunk = numba_generate_digits_core(
            BYTES_PER_CHUNK, dimension, axis_array, axis_lengths, seed_array, seed_lengths, 
            f_coeffs_np, f_powers_np, f_constant, np.uint64(total_bytes_generated)
        )
        
        final_bytes_chunk = numba_generate_bytes_from_digits(s_values_chunk, np.uint64(chunk_counter), buffer_size, num_threads)
        bytes_to_yield = final_bytes_chunk.tobytes()
        
        remaining_bytes = total_target_length - total_bytes_generated
        if len(bytes_to_yield) > remaining_bytes:
            yield bytes_to_yield[:remaining_bytes]
            total_bytes_generated += remaining_bytes
        else:
            yield bytes_to_yield
            total_bytes_generated += len(bytes_to_yield)

        chunk_counter += 1
        if chunk_counter % 5 == 0:
            print(f"Generated {total_bytes_generated / (1024*1024):.2f} MB...")

def save_output(byte_generator: Generator[bytes, None, None], file_path: Path) -> None:
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as file:
            for byte_chunk in byte_generator:
                file.write(byte_chunk)
    except Exception as e:
        raise Exception(f"Error saving output file: {e}")

# ==============================================================================
# Main Pipeline
# ==============================================================================

def execute_pipeline(config_path: Path, project_root: Path):
    try:
        print(">> System startup, checking JIT compilation status...")
        overall_start_time = time.perf_counter()
        
        # Get script directory (Local Dir)
        script_dir = Path(__file__).resolve().parent
        config = read_config(config_path)

        # --- Smart Path Resolution Logic ---
        def resolve_smart_path(config_key: str, default_rel_path: str) -> Path:
            # If the config has a key, use it; otherwise use default.
            path_str = config.get(config_key, default_rel_path)
            
            # 1. Try relative to Project Root (Standard structure)
            p1 = project_root / path_str
            if p1.exists(): 
                return p1
            
            # 2. Try relative to Script Directory (Local structure)
            p2 = script_dir / Path(path_str).name
            if p2.exists(): 
                return p2
            
            # 3. Return p1 as default to show expected path in errors
            return p1

        # Resolve paths
        # Note: If entropy source is not defined in config, it remains None/Empty here.
        # But if defined, we expect it to be in 'data/' usually.
        prime_file_path = resolve_smart_path("prime", "data/example_prime.txt")
        save_file_path = resolve_smart_path("save", "output/output.txt")
        
        # --- Entropy Source Injection ---
        entropy_source_str = config.get("external_entropy_source")
        if entropy_source_str:
            # Resolve the path provided in JSON (e.g., "data/seed.bin")
            entropy_path = resolve_smart_path("external_entropy_source", entropy_source_str)
            external_seed_int = get_external_seed(entropy_path)
            random.seed(external_seed_int)
            print(f">> External entropy source injected: {entropy_path.resolve()}")
        else:
            # Fallback to OS-level CSPRNG
            random.seed(int.from_bytes(os.urandom(64), byteorder='big'))
            print(">> No external entropy source specified. Initializing with OS-level CSPRNG.")
        # ------------------------------
        
        primes = read_primes(prime_file_path)

        dimension = config["dimension"]
        target_length = config.get("decimal", 100)
        num_seeds = config.get("num_seeds", 1)
        buffer_size = config.get("buffer_size", 16)
        iteration_count = config["iteration_count"]
        enable_dim_check = config["enable_dim_check"]
        dim_verification_map = config["dim_verification_map"]

        block_length = config.get("s1_length", 256) * 256

        # --- Layer 1: Initialization ---
        print(">> Executing Layer 1 (Initialization)...")
        if enable_dim_check == 1:
            verify_dimension(dim_verification_map, 1, len(config["F1"]["coeffs"]))
            
        extractor_name = config.get("RandomnessExtractor", "sha256")
        prime_combinations = generate_prime_combinations(primes, len(config["F1"]["coeffs"]), dimension * 2)
        initial_entropy_seeds = compute_polynomial_values(prime_combinations, config["F1"], extractor_name)
        
        state_matrix = generate_initial_state_matrix(initial_entropy_seeds, block_length)
        if not state_matrix: raise Exception("Failed to generate valid base state matrix")

        # --- Intermediate Layers: F2 to F_N ---
        for layer in range(2, iteration_count + 1):
            f_config, resolved_name = resolve_polynomial_params(config, layer)
            print(f">> Executing Layer {layer} (Params: {resolved_name})...")
            
            if enable_dim_check == 1:
                verify_dimension(dim_verification_map, layer, len(f_config["coeffs"]))
                
            state_matrix = run_intermediate_layer(state_matrix, dimension, f_config, block_length)

        # --- Final Layer: Output Generation ---
        final_layer_index = iteration_count + 1
        f_final_config, resolved_name_final = resolve_polynomial_params(config, final_layer_index)
        print(f">> Executing Final Layer Output (Params: {resolved_name_final})...")
        
        if enable_dim_check == 1:
            verify_dimension(dim_verification_map, final_layer_index, len(f_final_config["coeffs"]))

        byte_stream = generate_byte_stream(state_matrix, dimension, f_final_config, num_seeds, target_length, buffer_size)
        
        save_output(byte_stream, save_file_path)
        print(f"Byte stream generated and saved to: {save_file_path.resolve()}")

        end_time = time.perf_counter()
        print(f"Total execution time: {round(end_time - overall_start_time, 5)} seconds")

    except Exception as e:
        print(f"\n[Error] Execution exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

def main():
    # Locate Project Root (One level up from src)
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    
    # Default config path
    default_config_path = project_root / "Main Processing" / "Config.json"
    
    parser = argparse.ArgumentParser(description="High-Dimensional Polynomial Randomness Framework (HDPRF)")
    parser.add_argument("-c", "--config", type=str, default=str(default_config_path), 
                        help="Path to configuration file (default: Config.json)")
    parser.add_argument("--profile", action="store_true", help="Enable cProfile performance analysis")
    
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        execute_pipeline(config_path, project_root)
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats(20)
        print(s.getvalue())
    else:
        execute_pipeline(config_path, project_root)

if __name__ == "__main__":
    main()