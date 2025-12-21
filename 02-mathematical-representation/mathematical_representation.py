"""
The Bloch Sphere - Concept 2: Mathematical Representation
========================================================
The θ and φ parameterization of qubit states.
"""

import numpy as np

def main():
    print("=" * 60)
    print("MATHEMATICAL REPRESENTATION OF QUBIT STATES")
    print("=" * 60)
    
    # General form
    print("\n1. GENERAL QUBIT STATE")
    print("-" * 40)
    print("Any single-qubit pure state can be written as:")
    print()
    print("  |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩")
    print()
    print("where:")
    print("  θ ∈ [0, π]   - polar (colatitude) angle")
    print("  φ ∈ [0, 2π)  - azimuthal angle")
    
    # Bloch vector
    print("\n2. THE BLOCH VECTOR")
    print("-" * 40)
    print("The angles define a point on a unit sphere:")
    print()
    print("  x = sin(θ)cos(φ)")
    print("  y = sin(θ)sin(φ)")
    print("  z = cos(θ)")
    print()
    print("Bloch vector: r⃗ = (x, y, z) with |r⃗| = 1 for pure states")
    
    # Functions
    def bloch_to_state(theta, phi):
        """Convert Bloch angles to state vector."""
        alpha = np.cos(theta / 2)
        beta = np.exp(1j * phi) * np.sin(theta / 2)
        return np.array([alpha, beta])
    
    def state_to_bloch_angles(state):
        """Convert state vector to Bloch angles."""
        alpha, beta = state[0], state[1]
        theta = 2 * np.arccos(np.abs(alpha))
        if np.abs(beta) > 1e-10:
            phi = np.angle(beta) - np.angle(alpha)
            phi = phi % (2 * np.pi)
        else:
            phi = 0
        return theta, phi
    
    def bloch_to_cartesian(theta, phi):
        """Convert Bloch angles to Cartesian coordinates."""
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return (x, y, z)
    
    # Examples
    print("\n3. EXAMPLES: ANGLES TO STATE")
    print("-" * 40)
    
    examples = [
        (0, 0, '|0⟩ (North pole)'),
        (np.pi, 0, '|1⟩ (South pole)'),
        (np.pi/2, 0, '|+⟩ (+X axis)'),
        (np.pi/2, np.pi, '|-⟩ (-X axis)'),
        (np.pi/2, np.pi/2, '|+i⟩ (+Y axis)'),
        (np.pi/2, 3*np.pi/2, '|-i⟩ (-Y axis)'),
        (np.pi/3, np.pi/4, 'Custom state')
    ]
    
    print(f"{'(θ, φ)':<20} {'State Vector':<30} {'Description'}")
    print("-" * 70)
    
    for theta, phi, desc in examples:
        state = bloch_to_state(theta, phi)
        theta_str = f"({theta/np.pi:.2f}π, {phi/np.pi:.2f}π)"
        state_str = f"[{state[0]:.3f}, {state[1]:.3f}]"
        print(f"{theta_str:<20} {state_str:<30} {desc}")
    
    # Cartesian coordinates
    print("\n4. CARTESIAN COORDINATES")
    print("-" * 40)
    
    print(f"{'(θ, φ)':<20} {'(x, y, z)':<25} {'Description'}")
    print("-" * 60)
    
    for theta, phi, desc in examples:
        coords = bloch_to_cartesian(theta, phi)
        theta_str = f"({theta/np.pi:.2f}π, {phi/np.pi:.2f}π)"
        coords_str = f"({coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f})"
        print(f"{theta_str:<20} {coords_str:<25} {desc}")
    
    # Why θ/2?
    print("\n5. WHY θ/2 IN THE STATE FORMULA?")
    print("-" * 40)
    print("We use θ/2 (not θ) because:")
    print()
    print("• Opposite points on sphere = orthogonal states")
    print("• |0⟩ and |1⟩ are at θ = 0 and θ = π")
    print("• They must satisfy ⟨0|1⟩ = 0")
    print()
    print("Verification:")
    state_0 = bloch_to_state(0, 0)
    state_1 = bloch_to_state(np.pi, 0)
    inner_product = np.vdot(state_0, state_1)
    print(f"  |0⟩ = {state_0}")
    print(f"  |1⟩ = {state_1}")
    print(f"  ⟨0|1⟩ = {inner_product:.6f} ≈ 0 ✓")
    
    # Amplitude vs Phase
    print("\n6. AMPLITUDE VS PHASE")
    print("-" * 40)
    print(f"{'Angle':<10} {'Controls':<20} {'Physical Meaning'}")
    print("-" * 50)
    print(f"{'θ':<10} {'Relative amplitude':<20} {'P(0) vs P(1)'}")
    print(f"{'φ':<10} {'Relative phase':<20} {'Interference pattern'}")
    
    # Measurement probabilities
    print("\n7. MEASUREMENT PROBABILITIES FROM θ")
    print("-" * 40)
    print("P(|0⟩) = cos²(θ/2)")
    print("P(|1⟩) = sin²(θ/2)")
    print()
    
    print(f"{'θ':<15} {'P(|0⟩)':<15} {'P(|1⟩)'}")
    print("-" * 45)
    
    for theta_deg in [0, 30, 45, 60, 90, 120, 150, 180]:
        theta = np.radians(theta_deg)
        p0 = np.cos(theta / 2) ** 2
        p1 = np.sin(theta / 2) ** 2
        print(f"{theta_deg}° = {theta/np.pi:.2f}π    {p0:.4f}         {p1:.4f}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: θ controls probability, φ controls phase.")
    print("Both are needed for a complete description of the qubit!")
    print("=" * 60)

if __name__ == "__main__":
    main()
