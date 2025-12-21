"""
The Bloch Sphere - Concept 9: Exercises
======================================
Practice problems with solutions.
"""

import numpy as np

def main():
    print("=" * 60)
    print("BLOCH SPHERE EXERCISES")
    print("=" * 60)
    
    # Helper functions
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    
    def state_to_bloch(state):
        rho = np.outer(state, np.conj(state))
        x = 2 * np.real(rho[0, 1])
        y = 2 * np.imag(rho[0, 1])
        z = np.real(rho[0, 0] - rho[1, 1])
        return (x, y, z)
    
    def bloch_to_state(theta, phi):
        return np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])
    
    # Exercise 1
    print("\n" + "=" * 60)
    print("EXERCISE 1: Bloch Vector Calculation")
    print("=" * 60)
    print("\nProblem: Find the Bloch vector for the state:")
    print("  |ψ⟩ = cos(π/6)|0⟩ + e^(iπ/4)sin(π/6)|1⟩")
    print()
    
    input("Press Enter to see the solution...")
    
    print("\nSOLUTION:")
    print("-" * 40)
    theta = np.pi / 3  # Since θ/2 = π/6
    phi = np.pi / 4
    
    print(f"From |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩:")
    print(f"  θ/2 = π/6, so θ = π/3")
    print(f"  φ = π/4")
    print()
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    print(f"Bloch vector coordinates:")
    print(f"  x = sin(θ)cos(φ) = sin(π/3)cos(π/4) = {x:.4f}")
    print(f"  y = sin(θ)sin(φ) = sin(π/3)sin(π/4) = {y:.4f}")
    print(f"  z = cos(θ) = cos(π/3) = {z:.4f}")
    print()
    print(f"Answer: r⃗ = ({x:.4f}, {y:.4f}, {z:.4f})")
    
    # Verify
    state = bloch_to_state(theta, phi)
    coords = state_to_bloch(state)
    print(f"\nVerification: {tuple(round(c, 4) for c in coords)}")
    
    # Exercise 2
    print("\n" + "=" * 60)
    print("EXERCISE 2: Gate Rotation")
    print("=" * 60)
    print("\nProblem: Show that Rz(π/2) rotates |+⟩ to |+i⟩")
    print()
    
    input("Press Enter to see the solution...")
    
    print("\nSOLUTION:")
    print("-" * 40)
    
    def Rz(theta):
        return np.array([
            [np.exp(-1j*theta/2), 0],
            [0, np.exp(1j*theta/2)]
        ])
    
    ket_plus = np.array([1, 1]) / np.sqrt(2)
    
    print("Rz(π/2) = [[e^(-iπ/4), 0], [0, e^(iπ/4)]]")
    print()
    print(f"|+⟩ = {np.round(ket_plus, 4)}")
    
    result = Rz(np.pi/2) @ ket_plus
    print(f"\nRz(π/2)|+⟩ = {np.round(result, 4)}")
    
    # Factor out global phase
    global_phase = result[0] / np.abs(result[0])
    normalized = result / global_phase
    print(f"\nRemoving global phase e^(-iπ/4):")
    print(f"  = {np.round(normalized, 4)}")
    
    ket_plus_i = np.array([1, 1j]) / np.sqrt(2)
    print(f"\n|+i⟩ = {np.round(ket_plus_i, 4)}")
    print(f"\nAre they equal (up to global phase)? {np.allclose(normalized, ket_plus_i)}")
    
    print("\nGeometric interpretation:")
    print(f"  |+⟩ Bloch coords: {state_to_bloch(ket_plus)}")
    print(f"  |+i⟩ Bloch coords: {tuple(round(c, 2) for c in state_to_bloch(ket_plus_i))}")
    print("  90° rotation about Z takes +X to +Y")
    
    # Exercise 3
    print("\n" + "=" * 60)
    print("EXERCISE 3: Density Matrix")
    print("=" * 60)
    print("\nProblem: Compute the density matrix of |-⟩ and find its Bloch vector")
    print()
    
    input("Press Enter to see the solution...")
    
    print("\nSOLUTION:")
    print("-" * 40)
    
    ket_minus = np.array([1, -1]) / np.sqrt(2)
    print(f"|-⟩ = [1/√2, -1/√2]^T = {np.round(ket_minus, 4)}")
    print()
    
    rho = np.outer(ket_minus, np.conj(ket_minus))
    print("ρ = |-⟩⟨-| =")
    print(np.round(rho, 4))
    print()
    
    x = 2 * np.real(rho[0, 1])
    y = 2 * np.imag(rho[0, 1])
    z = np.real(rho[0, 0] - rho[1, 1])
    
    print("Extracting Bloch vector:")
    print(f"  x = 2·Re(ρ₀₁) = 2·({rho[0,1].real:.4f}) = {x:.4f}")
    print(f"  y = 2·Im(ρ₀₁) = 2·({rho[0,1].imag:.4f}) = {y:.4f}")
    print(f"  z = ρ₀₀ - ρ₁₁ = {rho[0,0].real:.4f} - {rho[1,1].real:.4f} = {z:.4f}")
    print()
    print(f"Answer: r⃗ = ({x:.1f}, {y:.1f}, {z:.1f}) — the -X axis!")
    
    # Exercise 4
    print("\n" + "=" * 60)
    print("EXERCISE 4: Sequential Gates")
    print("=" * 60)
    print("\nProblem: Trace the path of |0⟩ under H then Z on the Bloch sphere")
    print()
    
    input("Press Enter to see the solution...")
    
    print("\nSOLUTION:")
    print("-" * 40)
    
    ket_0 = np.array([1, 0])
    
    print("Step 1: Start at |0⟩")
    print(f"  Bloch coords: {state_to_bloch(ket_0)}")
    print()
    
    state_after_H = H @ ket_0
    print("Step 2: Apply H (180° rotation about (X+Z)/√2)")
    print(f"  H|0⟩ = |+⟩")
    print(f"  Bloch coords: {tuple(round(c, 2) for c in state_to_bloch(state_after_H))}")
    print()
    
    state_after_Z = Z @ state_after_H
    print("Step 3: Apply Z (180° rotation about Z)")
    print(f"  Z|+⟩ = |-⟩")
    print(f"  Bloch coords: {tuple(round(c, 2) for c in state_to_bloch(state_after_Z))}")
    print()
    
    print("Path: North pole → +X axis → -X axis")
    
    # Exercise 5
    print("\n" + "=" * 60)
    print("EXERCISE 5: Global Phase")
    print("=" * 60)
    print("\nProblem: Why doesn't global phase change the Bloch sphere position?")
    print()
    
    input("Press Enter to see the solution...")
    
    print("\nSOLUTION:")
    print("-" * 40)
    print("Consider |ψ⟩ and e^(iγ)|ψ⟩")
    print()
    print("The density matrix for e^(iγ)|ψ⟩ is:")
    print("  ρ' = (e^(iγ)|ψ⟩)(e^(-iγ)⟨ψ|)")
    print("     = e^(iγ)e^(-iγ)|ψ⟩⟨ψ|")
    print("     = |ψ⟩⟨ψ|")
    print("     = ρ")
    print()
    print("Since ρ is unchanged, and Bloch vector is derived from ρ:")
    print("  x = Tr(ρσx), y = Tr(ρσy), z = Tr(ρσz)")
    print()
    print("The Bloch vector (x, y, z) remains the same!")
    print()
    
    # Numerical verification
    gamma = np.pi / 3
    state = np.array([1, 1]) / np.sqrt(2)
    state_phased = np.exp(1j * gamma) * state
    
    print("Numerical verification:")
    print(f"  |ψ⟩ Bloch: {tuple(round(c, 4) for c in state_to_bloch(state))}")
    print(f"  e^(iπ/3)|ψ⟩ Bloch: {tuple(round(c, 4) for c in state_to_bloch(state_phased))}")
    
    # Exercise 6
    print("\n" + "=" * 60)
    print("EXERCISE 6: Measurement Probability")
    print("=" * 60)
    print("\nProblem: A qubit has Bloch vector (0.6, 0, 0.8).")
    print("What is P(|0⟩)?")
    print()
    
    input("Press Enter to see the solution...")
    
    print("\nSOLUTION:")
    print("-" * 40)
    print("For Z-basis measurement, P(|0⟩) depends on z-coordinate:")
    print()
    print("  P(|0⟩) = (1 + z)/2")
    print(f"         = (1 + 0.8)/2")
    print(f"         = 0.9")
    print(f"         = 90%")
    print()
    print("Alternatively, from z = cos(θ):")
    z = 0.8
    theta = np.arccos(z)
    p0 = np.cos(theta/2)**2
    print(f"  θ = arccos(0.8) = {theta:.4f} rad")
    print(f"  P(|0⟩) = cos²(θ/2) = {p0:.4f} = {p0*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("CONGRATULATIONS! You've completed all exercises!")
    print("=" * 60)

if __name__ == "__main__":
    main()
