import os
import sys
import numpy as np
import random
from collections import deque

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from exotic_geometry_framework import GeometryAnalyzer, delay_embed

def load_text_as_bytes(filepath, limit=50000):
    """Load text file and convert to byte array."""
    with open(filepath, 'rb') as f:
        data = f.read()
    
    # Simple header stripping: if we see the Project Gutenberg start, skip past it.
    # Otherwise just use the data.
    start_marker = b'*** START OF THE PROJECT GUTENBERG'
    if start_marker in data:
        parts = data.split(start_marker, 1)
        if len(parts) > 1:
            # Skip the marker line itself roughly (first 1000 chars after marker usually contain title block)
            # Just take the last part
            data = parts[1]
            # Try to skip the title block if another *** is nearby
            if b'***' in data[:100]:
                 data = data.split(b'***', 1)[-1]
    
    # Clean newlines and extra whitespace to standardize
    try:
        text = data.decode('utf-8', errors='ignore')
    except:
        text = str(data)
        
    text = ' '.join(text.split())
    
    # Return as bytes
    return np.frombuffer(text.encode('utf-8')[:limit], dtype=np.uint8)

def generate_synthetic_ai_text(length=50000):
    """
    Generates text that mimics 'bad' AI:
    - Highly repetitive structure
    - Frequent transition words
    - Bland vocabulary
    - Consistent sentence length
    """
    transitions = [
        "Furthermore,", "In addition,", "Moreover,", "Consequently,", 
        "It is important to note that", "As a result,", "However,", 
        "On the other hand,", "In conclusion,", "To summarize,"
    ]
    
    subjects = [
        "the utilization of artificial intelligence", "the implementation of robust protocols", 
        "the optimization of key performance indicators", "the strategic alignment of resources",
        "the facilitation of seamless integration", "the enhancement of user experience"
    ]
    
    verbs = [
        "demonstrates", "illustrates", "exemplifies", "signifies", 
        "indicates", "suggests", "highlights", "emphasizes", "underscores"
    ]
    
    objects = [
        "a significant improvement in efficiency.", "a reduction in operational costs.",
        "a paradigm shift in the industry.", "the potential for future scalability.",
        "the necessity for comprehensive analysis.", "the value of data-driven decision making."
    ]
    
    text = []
    current_len = 0
    
    while current_len < length:
        sentence = f"{random.choice(transitions)} {random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)}"
        text.append(sentence)
        current_len += len(sentence) + 1
        
    full_text = " ".join(text)
    return np.frombuffer(full_text.encode('utf-8')[:length], dtype=np.uint8)

def run_investigation():
    print("=== EXOTIC GEOMETRY: AI vs HUMAN TEXT DETECTION ===")
    
    # 1. Load Data
    human_path = "data/ai_text/human_alice.txt"
    if not os.path.exists(human_path):
        print(f"Error: {human_path} not found.")
        return

    print("Loading human text...")
    human_data = load_text_as_bytes(human_path)
    print(f"Human data size: {len(human_data)} bytes")
    
    print("Generating synthetic AI text...")
    ai_data = generate_synthetic_ai_text(len(human_data))
    print(f"AI data size: {len(ai_data)} bytes")
    
    # 2. Setup Analyzer
    analyzer = GeometryAnalyzer()
    analyzer.add_all_geometries()
    
    # 3. Analyze Human
    print("\nAnalyzing Human Text (Alice in Wonderland)...")
    human_results = analyzer.analyze(human_data)
    
    # 4. Analyze AI
    print("Analyzing Synthetic AI Text...")
    ai_results = analyzer.analyze(ai_data)
    
    # 5. Compare
    print("\n=== RESULTS: SIGNIFICANT DIFFERENCES ===")
    print("Metric | Human Val | AI Val | Diff | % Diff")
    print("-" * 60)
    
    significant_count = 0
    
    # Flatten results for comparison
    human_metrics = {}
    # FIX: Iterate over to_dict() results
    for g_name, metrics in human_results.to_dict().items():
        for m_name, val in metrics.items():
            human_metrics[f"{g_name}:{m_name}"] = val
            
    ai_metrics = {}
    for g_name, metrics in ai_results.to_dict().items():
        for m_name, val in metrics.items():
            ai_metrics[f"{g_name}:{m_name}"] = val
            
    # Compare
    keys = sorted(human_metrics.keys())
    for k in keys:
        h_val = human_metrics.get(k, 0)
        a_val = ai_metrics.get(k, 0)
        
        # Avoid division by zero
        denom = abs(h_val) + abs(a_val)
        if denom == 0:
            continue
            
        diff = abs(h_val - a_val)
        rel_diff = diff / (denom / 2) # Relative difference to mean
        
        # Threshold for "interesting" difference (heuristic, not statistical significance yet)
        if rel_diff > 0.5: # 50% difference
            print(f"{k:<40} | {h_val:8.4f} | {a_val:8.4f} | {diff:8.4f} | {rel_diff*100:3.0f}%")
            significant_count += 1
            
    print(f"\nTotal metrics with >50% difference: {significant_count}")
    
    # 6. Save results?
    # For now just print.

if __name__ == "__main__":
    run_investigation()
