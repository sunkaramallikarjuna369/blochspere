"""
The Bloch Sphere - Concept 10: Summary
=====================================
Key takeaways and quick reference.
"""

import numpy as np

def main():
    print("=" * 60)
    print("THE BLOCH SPHERE - COMPLETE SUMMARY")
    print("=" * 60)
    
    # Quick reference code
    print("\n" + "=" * 60)
    print("QUICK REFERENCE CODE")
    print("=" * 60)
    
    # Basic states
    print("\n1. BASIC STATES")
    print("-" * 40)
    
    ket_0 = np.array([1, 0])
    ket_1 = np.array([0, 1])
    ket_plus = np.array([1, 1]) / np.sqrt(2)
    ket_minus = np.array([1, -1]) / np.sqrt(2)
    ket_plus_i = np.array([1, 1j]) / np.sqrt(2)
    ket_minus_i = np.array([1, -1j]) / np.sqrt(2)
    
    print(f"|0⟩ = {ket_0}")
    print(f"|1⟩ = {ket_1}")
    print(f"|+⟩ = {np.round(ket_plus, 4)}")
    print(f"|-⟩ = {np.round(ket_minus, 4)}")
    print(f"|+i⟩ = {np.round(ket_plus_i, 4)}")
    print(f"|-i⟩ = {np.round(ket_minus_i, 4)}")
    
    # Pauli matrices
    print("\n2. PAULI MATRICES")
    print("-" * 40)
    
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    
    print("I = [[1,0],[0,1]]")
    print("X = [[0,1],[1,0]]")
    print("Y = [[0,-i],[i,0]]")
    print("Z = [[1,0],[0,-1]]")
    print("H = [[1,1],[1,-1]]/√2")
    
    # Bloch vector functions
    print("\n3. BLOCH VECTOR FUNCTIONS")
    print("-" * 40)
    
    def state_to_bloch(state):
        """Convert state vector to Bloch coordinates."""
        rho = np.outer(state, np.conj(state))
        x = np.real(np.trace(rho @ X))
        y = np.real(np.trace(rho @ Y))
        z = np.real(np.trace(rho @ Z))
        return (x, y, z)
    
    def bloch_to_state(theta, phi):
        """Convert Bloch angles to state vector."""
        return np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])
    
    def bloch_to_density(x, y, z):
        """Convert Bloch vector to density matrix."""
        return 0.5 * (I + x*X + y*Y + z*Z)
    
    def purity(rho):
        """Calculate purity Tr(ρ²)."""
        return np.real(np.trace(rho @ rho))
    
    print("state_to_bloch(state) → (x, y, z)")
    print("bloch_to_state(theta, phi) → state vector")
    print("bloch_to_density(x, y, z) → density matrix")
    print("purity(rho) → Tr(ρ²)")
    
    # Key coordinates
    print("\n4. KEY COORDINATES")
    print("-" * 40)
    
    states = {
        '|0⟩': ket_0,
        '|1⟩': ket_1,
        '|+⟩': ket_plus,
        '|-⟩': ket_minus,
        '|+i⟩': ket_plus_i,
        '|-i⟩': ket_minus_i
    }
    
    print(f"{'State':<8} {'(x, y, z)':<20} {'Location'}")
    print("-" * 45)
    for name, state in states.items():
        coords = state_to_bloch(state)
        coords_str = f"({coords[0]:4.1f}, {coords[1]:4.1f}, {coords[2]:4.1f})"
        location = {
            '|0⟩': 'North pole',
            '|1⟩': 'South pole',
            '|+⟩': '+X axis',
            '|-⟩': '-X axis',
            '|+i⟩': '+Y axis',
            '|-i⟩': '-Y axis'
        }[name]
        print(f"{name:<8} {coords_str:<20} {location}")
    
    # Gates as rotations
    print("\n5. GATES AS ROTATIONS")
    print("-" * 40)
    print(f"{'Gate':<8} {'Rotation'}")
    print("-" * 30)
    print(f"{'X':<8} {'180° about X-axis'}")
    print(f"{'Y':<8} {'180° about Y-axis'}")
    print(f"{'Z':<8} {'180° about Z-axis'}")
    print(f"{'H':<8} {'180° about (X+Z)/√2'}")
    print(f"{'S':<8} {'90° about Z-axis'}")
    print(f"{'T':<8} {'45° about Z-axis'}")
    
    # Measurement formulas
    print("\n6. MEASUREMENT FORMULAS")
    print("-" * 40)
    print("Z-basis:")
    print("  P(|0⟩) = cos²(θ/2) = (1+z)/2")
    print("  P(|1⟩) = sin²(θ/2) = (1-z)/2")
    print()
    print("X-basis:")
    print("  P(|+⟩) = (1+x)/2")
    print("  P(|-⟩) = (1-x)/2")
    print()
    print("Y-basis:")
    print("  P(|+i⟩) = (1+y)/2")
    print("  P(|-i⟩) = (1-y)/2")
    
    # Pure vs mixed
    print("\n7. PURE VS MIXED STATES")
    print("-" * 40)
    print("Pure states:  |r⃗| = 1 (on surface)")
    print("Mixed states: |r⃗| < 1 (inside)")
    print("Max mixed:    r⃗ = 0 (center)")
    print()
    print("Purity: Tr(ρ²) = ½(1 + |r⃗|²)")
    
    # Intuitive summary
    print("\n8. INTUITIVE SUMMARY")
    print("-" * 40)
    print(f"{'Concept':<25} {'Bloch Interpretation'}")
    print("-" * 55)
    print(f"{'Relative phase':<25} {'Longitude (φ)'}")
    print(f"{'Superposition degree':<25} {'Latitude (θ)'}")
    print(f"{'Measurement probability':<25} {'Distance from poles'}")
    print(f"{'Orthogonal states':<25} {'Antipodal points'}")
    print(f"{'Pure vs mixed':<25} {'Surface vs interior'}")
    print(f"{'Quantum gate':<25} {'3D rotation'}")
    
    # Demonstration
    print("\n" + "=" * 60)
    print("DEMONSTRATION")
    print("=" * 60)
    
    print("\nCreating and manipulating a qubit state:")
    print("-" * 40)
    
    # Create state
    theta, phi = np.pi/3, np.pi/4
    state = bloch_to_state(theta, phi)
    print(f"1. Create state with θ=π/3, φ=π/4:")
    print(f"   State: {np.round(state, 4)}")
    print(f"   Bloch: {tuple(round(c, 4) for c in state_to_bloch(state))}")
    
    # Apply gate
    state_after_H = H @ state
    print(f"\n2. Apply Hadamard gate:")
    print(f"   State: {np.round(state_after_H, 4)}")
    print(f"   Bloch: {tuple(round(c, 4) for c in state_to_bloch(state_after_H))}")
    
    # Measurement probabilities
    p0 = np.abs(state_after_H[0])**2
    p1 = np.abs(state_after_H[1])**2
    print(f"\n3. Z-basis measurement probabilities:")
    print(f"   P(|0⟩) = {p0:.4f}")
    print(f"   P(|1⟩) = {p1:.4f}")
    
    # Simulate measurement
    np.random.seed(42)
    outcomes = np.random.choice([0, 1], size=1000, p=[p0, p1])
    print(f"\n4. Simulated measurement (1000 shots):")
    print(f"   |0⟩: {np.sum(outcomes==0)} ({np.mean(outcomes==0)*100:.1f}%)")
    print(f"   |1⟩: {np.sum(outcomes==1)} ({np.mean(outcomes==1)*100:.1f}%)")
    
    # What's next
    print("\n" + "=" * 60)
    print("WHAT'S NEXT?")
    print("=" * 60)
    print()
    print("Understanding the Bloch sphere prepares you for:")
    print()
    print("• Multi-qubit systems — Tensor products and entanglement")
    print("• Quantum entanglement — Bell states, non-local correlations")
    print("• Quantum circuits — Building quantum algorithms")
    print("• Quantum tomography — Reconstructing states from measurements")
    print("• Quantum error correction — Protecting quantum information")
    
    print("\n" + "=" * 60)
    print("CONGRATULATIONS!")
    print("You now have the geometric intuition needed to")
    print("understand single-qubit quantum mechanics!")
    print("=" * 60)

if __name__ == "__main__":
    main()
