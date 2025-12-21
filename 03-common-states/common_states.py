"""
The Bloch Sphere - Concept 3: Common States
==========================================
The six cardinal points of the Bloch sphere.
"""

import numpy as np

def main():
    print("=" * 60)
    print("COMMON STATES ON THE BLOCH SPHERE")
    print("=" * 60)
    
    # Define states
    states = {
        '|0⟩': np.array([1, 0]),
        '|1⟩': np.array([0, 1]),
        '|+⟩': np.array([1, 1]) / np.sqrt(2),
        '|-⟩': np.array([1, -1]) / np.sqrt(2),
        '|+i⟩': np.array([1, 1j]) / np.sqrt(2),
        '|-i⟩': np.array([1, -1j]) / np.sqrt(2)
    }
    
    def state_to_bloch(state):
        """Convert state to Bloch coordinates."""
        rho = np.outer(state, np.conj(state))
        x = 2 * np.real(rho[0, 1])
        y = 2 * np.imag(rho[0, 1])
        z = np.real(rho[0, 0] - rho[1, 1])
        return (x, y, z)
    
    # Computational basis
    print("\n1. COMPUTATIONAL BASIS (Z-axis)")
    print("-" * 40)
    print("|0⟩ = [1, 0]^T")
    print("  θ = 0, φ = 0")
    print("  Bloch vector: (0, 0, 1) — North Pole")
    print()
    print("|1⟩ = [0, 1]^T")
    print("  θ = π, φ = 0")
    print("  Bloch vector: (0, 0, -1) — South Pole")
    
    # Hadamard basis
    print("\n2. HADAMARD BASIS (X-axis)")
    print("-" * 40)
    print("|+⟩ = (|0⟩ + |1⟩)/√2 = [1, 1]^T/√2")
    print("  θ = π/2, φ = 0")
    print("  Bloch vector: (1, 0, 0) — +X Axis")
    print()
    print("|-⟩ = (|0⟩ - |1⟩)/√2 = [1, -1]^T/√2")
    print("  θ = π/2, φ = π")
    print("  Bloch vector: (-1, 0, 0) — -X Axis")
    
    # Circular basis
    print("\n3. CIRCULAR BASIS (Y-axis)")
    print("-" * 40)
    print("|+i⟩ = (|0⟩ + i|1⟩)/√2 = [1, i]^T/√2")
    print("  θ = π/2, φ = π/2")
    print("  Bloch vector: (0, 1, 0) — +Y Axis")
    print()
    print("|-i⟩ = (|0⟩ - i|1⟩)/√2 = [1, -i]^T/√2")
    print("  θ = π/2, φ = 3π/2")
    print("  Bloch vector: (0, -1, 0) — -Y Axis")
    
    # Complete reference table
    print("\n4. COMPLETE REFERENCE TABLE")
    print("-" * 70)
    print(f"{'State':<8} {'Vector':<20} {'(θ, φ)':<15} {'Bloch (x,y,z)'}")
    print("-" * 70)
    
    state_info = [
        ('|0⟩', '[1, 0]', '(0, 0)', states['|0⟩']),
        ('|1⟩', '[0, 1]', '(π, 0)', states['|1⟩']),
        ('|+⟩', '[1, 1]/√2', '(π/2, 0)', states['|+⟩']),
        ('|-⟩', '[1, -1]/√2', '(π/2, π)', states['|-⟩']),
        ('|+i⟩', '[1, i]/√2', '(π/2, π/2)', states['|+i⟩']),
        ('|-i⟩', '[1, -i]/√2', '(π/2, 3π/2)', states['|-i⟩'])
    ]
    
    for name, vec, angles, state in state_info:
        coords = state_to_bloch(state)
        coords_str = f"({coords[0]:4.1f}, {coords[1]:4.1f}, {coords[2]:4.1f})"
        print(f"{name:<8} {vec:<20} {angles:<15} {coords_str}")
    
    # Orthogonality
    print("\n5. ORTHOGONALITY CHECK")
    print("-" * 40)
    print("Antipodal points on the sphere are orthogonal states:")
    print()
    
    pairs = [
        ('|0⟩', '|1⟩'),
        ('|+⟩', '|-⟩'),
        ('|+i⟩', '|-i⟩')
    ]
    
    for s1, s2 in pairs:
        inner = np.vdot(states[s1], states[s2])
        print(f"⟨{s1[1:-1]}|{s2[1:-1]}⟩ = {inner:.6f}")
    
    # Measurement in different bases
    print("\n6. MEASUREMENT IN DIFFERENT BASES")
    print("-" * 40)
    
    test_state = states['|+⟩']
    print(f"Test state: |+⟩ = {np.round(test_state, 4)}")
    print()
    
    print("Z-basis measurement:")
    p0 = np.abs(np.vdot(states['|0⟩'], test_state))**2
    p1 = np.abs(np.vdot(states['|1⟩'], test_state))**2
    print(f"  P(|0⟩) = {p0:.4f}")
    print(f"  P(|1⟩) = {p1:.4f}")
    
    print("\nX-basis measurement:")
    p_plus = np.abs(np.vdot(states['|+⟩'], test_state))**2
    p_minus = np.abs(np.vdot(states['|-⟩'], test_state))**2
    print(f"  P(|+⟩) = {p_plus:.4f}")
    print(f"  P(|-⟩) = {p_minus:.4f}")
    
    print("\nY-basis measurement:")
    p_plus_i = np.abs(np.vdot(states['|+i⟩'], test_state))**2
    p_minus_i = np.abs(np.vdot(states['|-i⟩'], test_state))**2
    print(f"  P(|+i⟩) = {p_plus_i:.4f}")
    print(f"  P(|-i⟩) = {p_minus_i:.4f}")
    
    # Visualization hint
    print("\n7. GEOMETRIC RELATIONSHIPS")
    print("-" * 40)
    print("• Z-axis: Computational basis (|0⟩, |1⟩)")
    print("• X-axis: Hadamard basis (|+⟩, |-⟩)")
    print("• Y-axis: Circular basis (|+i⟩, |-i⟩)")
    print()
    print("• Equator (z=0): Equal superpositions of |0⟩ and |1⟩")
    print("• Poles: Pure |0⟩ or |1⟩ states")
    print("• Antipodal points: Orthogonal states")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: The six cardinal states form three mutually")
    print("unbiased bases — each basis is maximally uncertain in the others!")
    print("=" * 60)

if __name__ == "__main__":
    main()
