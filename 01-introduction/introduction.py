"""
The Bloch Sphere - Concept 1: Introduction
==========================================
Why we need the Bloch representation for quantum states.
"""

import numpy as np

def main():
    print("=" * 60)
    print("THE BLOCH SPHERE - INTRODUCTION")
    print("=" * 60)
    
    # Classical bit
    print("\n1. THE CLASSICAL BIT")
    print("-" * 40)
    print("A classical bit can only be in one of two states:")
    print("  Bit = 0  or  Bit = 1")
    print("\nNo superposition, no in-between states.")
    
    # Quantum bit
    print("\n2. THE QUANTUM BIT (QUBIT)")
    print("-" * 40)
    print("A qubit can exist in a superposition:")
    print("  |ψ⟩ = α|0⟩ + β|1⟩")
    print("\nwhere α, β ∈ ℂ and |α|² + |β|² = 1")
    
    # Example qubit states
    print("\n3. EXAMPLE QUBIT STATES")
    print("-" * 40)
    
    # |0⟩ state
    ket_0 = np.array([1, 0])
    print(f"|0⟩ = {ket_0}")
    print(f"  P(0) = |α|² = {np.abs(ket_0[0])**2:.2f}")
    print(f"  P(1) = |β|² = {np.abs(ket_0[1])**2:.2f}")
    
    # |+⟩ state (equal superposition)
    ket_plus = np.array([1, 1]) / np.sqrt(2)
    print(f"\n|+⟩ = (|0⟩ + |1⟩)/√2 = {np.round(ket_plus, 4)}")
    print(f"  P(0) = |α|² = {np.abs(ket_plus[0])**2:.2f}")
    print(f"  P(1) = |β|² = {np.abs(ket_plus[1])**2:.2f}")
    
    # Complex superposition
    ket_complex = np.array([1, 1j]) / np.sqrt(2)
    print(f"\n|+i⟩ = (|0⟩ + i|1⟩)/√2 = {np.round(ket_complex, 4)}")
    print(f"  P(0) = |α|² = {np.abs(ket_complex[0])**2:.2f}")
    print(f"  P(1) = |β|² = {np.abs(ket_complex[1])**2:.2f}")
    
    # Why we need Bloch sphere
    print("\n4. WHY WE NEED THE BLOCH SPHERE")
    print("-" * 40)
    print("A qubit state has 2 complex numbers = 4 real parameters")
    print("But we can reduce this:")
    print("  - Normalization: |α|² + |β|² = 1  →  removes 1 parameter")
    print("  - Global phase: e^(iγ)|ψ⟩ is physically identical  →  removes 1 parameter")
    print("\nResult: Only 2 real parameters needed!")
    print("These are θ (theta) and φ (phi) on the Bloch sphere.")
    
    # Parameter counting
    print("\n5. PARAMETER COUNTING")
    print("-" * 40)
    print(f"{'Description':<35} {'Parameters'}")
    print("-" * 50)
    print(f"{'Two complex numbers (α, β)':<35} {'4 real'}")
    print(f"{'After normalization':<35} {'3 real'}")
    print(f"{'After removing global phase':<35} {'2 real (θ, φ)'}")
    
    # Bloch sphere representation
    print("\n6. BLOCH SPHERE REPRESENTATION")
    print("-" * 40)
    print("General qubit state:")
    print("  |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩")
    print()
    print("where:")
    print("  θ ∈ [0, π]   - polar angle (latitude)")
    print("  φ ∈ [0, 2π)  - azimuthal angle (longitude)")
    print()
    print("Bloch vector: r⃗ = (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ))")
    
    # Demonstrate the mapping
    print("\n7. MAPPING STATES TO THE SPHERE")
    print("-" * 40)
    
    def state_to_bloch(state):
        """Convert state vector to Bloch coordinates."""
        rho = np.outer(state, np.conj(state))
        x = 2 * np.real(rho[0, 1])
        y = 2 * np.imag(rho[0, 1])
        z = np.real(rho[0, 0] - rho[1, 1])
        return (x, y, z)
    
    states = {
        '|0⟩': np.array([1, 0]),
        '|1⟩': np.array([0, 1]),
        '|+⟩': np.array([1, 1]) / np.sqrt(2),
        '|-⟩': np.array([1, -1]) / np.sqrt(2),
        '|+i⟩': np.array([1, 1j]) / np.sqrt(2),
        '|-i⟩': np.array([1, -1j]) / np.sqrt(2)
    }
    
    print(f"{'State':<10} {'Bloch Vector (x, y, z)'}")
    print("-" * 40)
    for name, state in states.items():
        coords = state_to_bloch(state)
        print(f"{name:<10} ({coords[0]:6.2f}, {coords[1]:6.2f}, {coords[2]:6.2f})")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: The Bloch sphere provides a complete 3D")
    print("visualization of all possible single-qubit pure states!")
    print("=" * 60)

if __name__ == "__main__":
    main()
