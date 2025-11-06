#!/usr/bin/env python3
"""
Demo script for 3-node harmonic validation
Emulates 3 nodes generating snapshots, computing aggregated CS, and attempting minting
"""

import sys
import os
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openagi.harmonic_validation import make_snapshot, recursive_validate


def synth_signal(freq, phase, duration=0.5, sample_rate=2048):
    """Generate a synthetic signal"""
    t = np.linspace(0, duration, int(duration * sample_rate))
    x = np.sin(2 * np.pi * freq * t + phase)
    return t, x


def main():
    print("🔬 Demo de Validación Armónica de 3 Nodos")
    print("=" * 50)
    
    # Generate time base
    t = np.linspace(0, 0.5, 2048)
    
    # Node A: coherent sine (50 Hz)
    a_vals = np.sin(2 * np.pi * 50 * t) + 0.01 * np.random.randn(len(t))
    
    # Node B: same frequency, same phase (coherent)
    b_vals = np.sin(2 * np.pi * 50 * t) + 0.01 * np.random.randn(len(t))
    
    # Node C: different frequency (incoherent)
    c_vals = np.sin(2 * np.pi * 60 * t + 0.2) + 0.1 * np.random.randn(len(t))
    
    # Create snapshots for each node
    print("📸 Generando snapshots de nodos...")
    snapshot_a = make_snapshot("node-A", t.tolist(), a_vals.tolist(), secret_key="keyA")
    snapshot_b = make_snapshot("node-B", t.tolist(), b_vals.tolist(), secret_key="keyB")
    snapshot_c = make_snapshot("node-C", t.tolist(), c_vals.tolist(), secret_key="keyC")
    
    # Create bundle for validation
    bundle = [snapshot_a, snapshot_b, snapshot_c]
    
    # Perform recursive validation
    print("🔄 Validando coherencia armónica...")
    is_valid, proof_bundle = recursive_validate(bundle, threshold=0.75)
    
    # Display results
    print(f"\n📊 Resultados de Validación:")
    if proof_bundle is not None:
        print(f"   Coherencia Agregada: {proof_bundle.aggregated_CS:.4f}")
        print(f"   Umbral Requerido: 0.75")
        print(f"   Validación: {'✅ APROBADA' if is_valid else '❌ RECHAZADA'}")
        
        # Decision based on coherence
        if proof_bundle.aggregated_CS >= 0.75:
            print("\n💰 Coherencia suficiente: se permite acuñar FLX.")
        else:
            print("\n🚫 Coherencia insuficiente: se rechaza la acuñación.")
    else:
        print("   Error en la validación: no se generó proof bundle")
        print("   Validación: ❌ RECHAZADA")
    
    # Show individual snapshot info
    print(f"\n📋 Información de Snapshots:")
    print(f"   Nodo A: CS={snapshot_a.CS:.4f}, Hash={snapshot_a.spectrum_hash[:8]}...")
    print(f"   Nodo B: CS={snapshot_b.CS:.4f}, Hash={snapshot_b.spectrum_hash[:8]}...")
    print(f"   Nodo C: CS={snapshot_c.CS:.4f}, Hash={snapshot_c.spectrum_hash[:8]}...")
    
    print(f"\n🎯 Demo completada exitosamente!")


if __name__ == "__main__":
    main()