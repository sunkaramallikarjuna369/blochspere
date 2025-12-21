"""
The Bloch Sphere - Concept 8: Mixed States
=========================================
Points inside the Bloch sphere.
"""

import numpy as np

def main():
    print("=" * 60)
    print("MIXED STATES: INSIDE THE BLOCH SPHERE")
    print("=" * 60)
    
    # Pauli matrices
    I = np.eye(2)
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    
    # Define states
    ket_0 = np.array([1, 0])
    ket_1 = np.array([0, 1])
    
    def bloch_to_density(x, y, z):
        """Convert Bloch vector to density matrix."""
        return 0.5 * (I + x*sigma_x + y*sigma_y + z*sigma_z)
    
    def density_to_bloch(rho):
        """Extract Bloch vector from density matrix."""
        x = np.real(np.trace(rho @ sigma_x))
        y = np.real(np.trace(rho @ sigma_y))
        z = np.real(np.trace(rho @ sigma_z))
        return (x, y, z)
    
    def purity(rho):
        """Calculate purity Tr(ρ²)."""
        return np.real(np.trace(rho @ rho))
    
    def bloch_length(rho):
        """Calculate |r| from density matrix."""
        x, y, z = density_to_bloch(rho)
        return np.sqrt(x**2 + y**2 + z**2)
    
    # Pure vs mixed
    print("\n1. PURE VS MIXED STATES")
    print("-" * 40)
    print("PURE STATES:")
    print("  • Lie on the SURFACE of the Bloch sphere")
    print("  • |r⃗| = 1")
    print("  • Can be written as |ψ⟩⟨ψ|")
    print("  • Maximum quantum coherence")
    print()
    print("MIXED STATES:")
    print("  • Lie INSIDE the Bloch sphere")
    print("  • |r⃗| < 1")
    print("  • Statistical mixture of pure states")
    print("  • Reduced quantum coherence")
    
    # Creating mixed states
    print("\n2. CREATING MIXED STATES")
    print("-" * 40)
    print("A mixed state can be created by:")
    print()
    print("1. Classical uncertainty:")
    print("   ρ = p|0⟩⟨0| + (1-p)|1⟩⟨1|")
    print()
    
    def create_classical_mixture(p):
        """Create mixture of |0⟩ and |1⟩ with probability p."""
        rho_0 = np.outer(ket_0, ket_0)
        rho_1 = np.outer(ket_1, ket_1)
        return p * rho_0 + (1 - p) * rho_1
    
    print(f"{'p':<10} {'|r⃗|':<10} {'Tr(ρ²)':<10} {'State Type'}")
    print("-" * 45)
    
    for p in [1.0, 0.9, 0.75, 0.5, 0.25, 0.0]:
        rho = create_classical_mixture(p)
        r = bloch_length(rho)
        pur = purity(rho)
        state_type = 'Pure |0⟩' if p == 1 else ('Pure |1⟩' if p == 0 else ('Max mixed' if p == 0.5 else 'Mixed'))
        print(f"{p:<10.2f} {r:<10.4f} {pur:<10.4f} {state_type}")
    
    # Maximally mixed state
    print("\n3. MAXIMALLY MIXED STATE")
    print("-" * 40)
    print("At the center of the Bloch sphere (r⃗ = 0):")
    print()
    
    rho_max_mixed = I / 2
    print("ρ = I/2 =")
    print(rho_max_mixed)
    print()
    print(f"Bloch vector: {density_to_bloch(rho_max_mixed)}")
    print(f"Purity: Tr(ρ²) = {purity(rho_max_mixed):.4f}")
    print()
    print("This represents:")
    print("  • Complete ignorance about the qubit state")
    print("  • 50-50 mixture of |0⟩ and |1⟩")
    print("  • No quantum coherence")
    print("  • Minimum purity = 1/2")
    
    # Purity and coherence
    print("\n4. PURITY AND COHERENCE")
    print("-" * 40)
    print("The purity Tr(ρ²) measures 'quantumness':")
    print()
    print("  Tr(ρ²) = ½(1 + |r⃗|²)")
    print()
    print(f"{'|r⃗|':<10} {'Tr(ρ²)':<10} {'Description'}")
    print("-" * 40)
    print(f"{'1.0':<10} {'1.0':<10} {'Pure state (max coherence)'}")
    print(f"{'0.5':<10} {'0.625':<10} {'Partially mixed'}")
    print(f"{'0.0':<10} {'0.5':<10} {'Maximally mixed (no coherence)'}")
    
    # Sources of mixed states
    print("\n5. SOURCES OF MIXED STATES")
    print("-" * 40)
    print("Mixed states arise from:")
    print()
    print("1. CLASSICAL UNCERTAINTY")
    print("   We don't know which pure state was prepared")
    print()
    print("2. DECOHERENCE")
    print("   Interaction with environment destroys coherence")
    print()
    print("3. PARTIAL TRACE")
    print("   Tracing out part of an entangled system")
    
    # Decoherence simulation
    print("\n6. DECOHERENCE SIMULATION")
    print("-" * 40)
    print("Starting with |+⟩ state, applying dephasing:")
    print()
    
    # Start with |+⟩
    ket_plus = np.array([1, 1]) / np.sqrt(2)
    rho_pure = np.outer(ket_plus, ket_plus)
    
    print(f"{'Dephasing':<12} {'|r⃗|':<10} {'Tr(ρ²)':<10} {'x':<10} {'z'}")
    print("-" * 55)
    
    for gamma in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        # Dephasing reduces off-diagonal elements
        rho_dephased = rho_pure.copy()
        rho_dephased[0, 1] *= (1 - gamma)
        rho_dephased[1, 0] *= (1 - gamma)
        
        r = bloch_length(rho_dephased)
        pur = purity(rho_dephased)
        x, y, z = density_to_bloch(rho_dephased)
        
        print(f"{gamma:<12.1f} {r:<10.4f} {pur:<10.4f} {x:<10.4f} {z:.4f}")
    
    # Geometric interpretation
    print("\n7. GEOMETRIC INTERPRETATION")
    print("-" * 40)
    print("Distance from center = degree of quantum coherence")
    print()
    print("• Surface (|r⃗| = 1): Maximum coherence")
    print("• Inside (0 < |r⃗| < 1): Partial coherence")
    print("• Center (|r⃗| = 0): No coherence")
    print()
    print("Direction of r⃗ = average polarization")
    
    # Example: partially mixed state
    print("\n8. EXAMPLE: PARTIALLY MIXED STATE")
    print("-" * 40)
    
    # Create a state with |r⃗| = 0.6 pointing in +X direction
    r = 0.6
    rho = bloch_to_density(r, 0, 0)
    
    print(f"Bloch vector: ({r}, 0, 0)")
    print(f"Density matrix:")
    print(np.round(rho, 4))
    print()
    print(f"|r⃗| = {bloch_length(rho):.4f}")
    print(f"Purity = {purity(rho):.4f}")
    print()
    print("This could represent:")
    print(f"  • 80% |+⟩ and 20% |-⟩")
    print(f"  • Or a pure state that has undergone partial decoherence")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: Mixed states represent classical uncertainty")
    print("or decoherence. They live inside the Bloch sphere!")
    print("=" * 60)

if __name__ == "__main__":
    main()
