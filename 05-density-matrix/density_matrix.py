"""
The Bloch Sphere - Concept 5: Density Matrix
============================================
Bloch vector, Pauli matrices, and density operators.
"""

import numpy as np

def main():
    print("=" * 60)
    print("DENSITY MATRIX AND THE BLOCH SPHERE")
    print("=" * 60)
    
    # Pauli matrices
    print("\n1. THE PAULI MATRICES")
    print("-" * 40)
    
    I = np.eye(2)
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    
    print("Identity matrix I:")
    print(I)
    print()
    print("Pauli-X (σx):")
    print(sigma_x)
    print()
    print("Pauli-Y (σy):")
    print(sigma_y)
    print()
    print("Pauli-Z (σz):")
    print(sigma_z)
    
    # Density matrix formula
    print("\n2. DENSITY MATRIX FORMULA")
    print("-" * 40)
    print("Every single-qubit state can be written as:")
    print()
    print("  ρ = ½(I + r⃗ · σ⃗)")
    print("    = ½(I + x·σx + y·σy + z·σz)")
    print()
    print("where r⃗ = (x, y, z) is the Bloch vector")
    
    # Expanded form
    print("\n3. EXPANDED FORM")
    print("-" * 40)
    print("In matrix form:")
    print()
    print("  ρ = ½ | 1+z    x-iy |")
    print("       | x+iy   1-z  |")
    
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
    
    # Examples
    print("\n4. EXAMPLES")
    print("-" * 40)
    
    examples = [
        ((0, 0, 1), '|0⟩'),
        ((0, 0, -1), '|1⟩'),
        ((1, 0, 0), '|+⟩'),
        ((-1, 0, 0), '|-⟩'),
        ((0, 1, 0), '|+i⟩'),
        ((0, 0, 0), 'Maximally mixed')
    ]
    
    for (x, y, z), name in examples:
        rho = bloch_to_density(x, y, z)
        pur = purity(rho)
        print(f"\n{name} state: r⃗ = ({x}, {y}, {z})")
        print(f"Density matrix:")
        print(np.round(rho, 4))
        print(f"Purity: Tr(ρ²) = {pur:.4f}")
    
    # Extracting Bloch vector
    print("\n5. EXTRACTING BLOCH VECTOR FROM ρ")
    print("-" * 40)
    print("Given ρ, we can extract the Bloch vector:")
    print()
    print("  x = Tr(ρ σx) = 2 Re(ρ₀₁)")
    print("  y = Tr(ρ σy) = 2 Im(ρ₀₁)")
    print("  z = Tr(ρ σz) = ρ₀₀ - ρ₁₁")
    print()
    
    # Verify round-trip
    print("Verification (round-trip):")
    for (x, y, z), name in examples[:4]:
        rho = bloch_to_density(x, y, z)
        x2, y2, z2 = density_to_bloch(rho)
        print(f"  {name}: ({x},{y},{z}) → ρ → ({x2:.1f},{y2:.1f},{z2:.1f})")
    
    # Purity
    print("\n6. PURITY AND THE BLOCH VECTOR")
    print("-" * 40)
    print("The purity is related to the Bloch vector length:")
    print()
    print("  Tr(ρ²) = ½(1 + |r⃗|²)")
    print()
    print(f"{'|r⃗|':<10} {'Tr(ρ²)':<10} {'State Type'}")
    print("-" * 40)
    
    for r in [1.0, 0.8, 0.5, 0.2, 0.0]:
        pur = 0.5 * (1 + r**2)
        state_type = 'Pure' if r == 1 else ('Maximally mixed' if r == 0 else 'Mixed')
        print(f"{r:<10.2f} {pur:<10.4f} {state_type}")
    
    # Properties
    print("\n7. DENSITY MATRIX PROPERTIES")
    print("-" * 40)
    
    rho = bloch_to_density(0.5, 0.3, 0.4)
    print("For any valid density matrix ρ:")
    print()
    print(f"  1. Hermitian: ρ = ρ†")
    print(f"     Check: {np.allclose(rho, rho.conj().T)}")
    print()
    print(f"  2. Trace = 1: Tr(ρ) = 1")
    print(f"     Check: Tr(ρ) = {np.real(np.trace(rho)):.4f}")
    print()
    print(f"  3. Positive semi-definite: eigenvalues ≥ 0")
    eigenvalues = np.linalg.eigvalsh(rho)
    print(f"     Eigenvalues: {np.round(eigenvalues, 4)}")
    print(f"     All ≥ 0: {np.all(eigenvalues >= -1e-10)}")
    
    # Pure state condition
    print("\n8. PURE STATE CONDITION")
    print("-" * 40)
    print("A state is pure if and only if:")
    print("  • ρ² = ρ (idempotent)")
    print("  • Tr(ρ²) = 1")
    print("  • |r⃗| = 1")
    print()
    
    # Check for |+⟩
    rho_pure = bloch_to_density(1, 0, 0)
    print(f"|+⟩ state:")
    print(f"  ρ² = ρ? {np.allclose(rho_pure @ rho_pure, rho_pure)}")
    print(f"  Tr(ρ²) = {purity(rho_pure):.4f}")
    
    # Check for mixed state
    rho_mixed = bloch_to_density(0.5, 0, 0)
    print(f"\nMixed state (|r⃗| = 0.5):")
    print(f"  ρ² = ρ? {np.allclose(rho_mixed @ rho_mixed, rho_mixed)}")
    print(f"  Tr(ρ²) = {purity(rho_mixed):.4f}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: The density matrix provides a complete")
    print("description of quantum states, including mixed states!")
    print("=" * 60)

if __name__ == "__main__":
    main()
