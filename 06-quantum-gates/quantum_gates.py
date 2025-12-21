"""
The Bloch Sphere - Concept 6: Quantum Gates
==========================================
Rotations on the Bloch sphere.
"""

import numpy as np

def main():
    print("=" * 60)
    print("QUANTUM GATES AS ROTATIONS")
    print("=" * 60)
    
    # Define gates
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]])
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
    
    # Define states
    ket_0 = np.array([1, 0])
    ket_1 = np.array([0, 1])
    ket_plus = np.array([1, 1]) / np.sqrt(2)
    ket_minus = np.array([1, -1]) / np.sqrt(2)
    
    def state_to_bloch(state):
        """Convert state to Bloch coordinates."""
        rho = np.outer(state, np.conj(state))
        x = 2 * np.real(rho[0, 1])
        y = 2 * np.imag(rho[0, 1])
        z = np.real(rho[0, 0] - rho[1, 1])
        return (x, y, z)
    
    # Gates as rotations
    print("\n1. GATES AS ROTATIONS")
    print("-" * 40)
    print("Single-qubit unitary gates act as rotations:")
    print()
    print("  R_n(θ) = exp(-iθ n⃗·σ⃗/2)")
    print()
    print("This rotates the Bloch vector by angle θ about axis n⃗.")
    
    # Pauli gates
    print("\n2. PAULI GATES")
    print("-" * 40)
    print(f"{'Gate':<8} {'Matrix':<25} {'Rotation'}")
    print("-" * 55)
    print(f"{'X':<8} {'[[0,1],[1,0]]':<25} {'180° about X-axis'}")
    print(f"{'Y':<8} {'[[0,-i],[i,0]]':<25} {'180° about Y-axis'}")
    print(f"{'Z':<8} {'[[1,0],[0,-1]]':<25} {'180° about Z-axis'}")
    
    # Demonstrate Pauli gates
    print("\n3. PAULI GATE ACTIONS")
    print("-" * 40)
    
    print("\nX gate (bit flip):")
    for name, state in [('|0⟩', ket_0), ('|1⟩', ket_1), ('|+⟩', ket_plus)]:
        before = state_to_bloch(state)
        after = state_to_bloch(X @ state)
        print(f"  {name}: {before} → {tuple(round(x, 2) for x in after)}")
    
    print("\nY gate:")
    for name, state in [('|0⟩', ket_0), ('|1⟩', ket_1)]:
        before = state_to_bloch(state)
        after = state_to_bloch(Y @ state)
        print(f"  {name}: {before} → {tuple(round(x, 2) for x in after)}")
    
    print("\nZ gate (phase flip):")
    for name, state in [('|0⟩', ket_0), ('|+⟩', ket_plus), ('|-⟩', ket_minus)]:
        before = state_to_bloch(state)
        after = state_to_bloch(Z @ state)
        print(f"  {name}: {before} → {tuple(round(x, 2) for x in after)}")
    
    # Hadamard gate
    print("\n4. HADAMARD GATE")
    print("-" * 40)
    print("H = (X + Z)/√2")
    print("Rotation: 180° about (X+Z)/√2 axis")
    print()
    
    print("Hadamard gate actions:")
    for name, state in [('|0⟩', ket_0), ('|1⟩', ket_1), ('|+⟩', ket_plus)]:
        before = state_to_bloch(state)
        after = state_to_bloch(H @ state)
        print(f"  H{name}: {before} → {tuple(round(x, 2) for x in after)}")
    
    # Phase gates
    print("\n5. PHASE GATES")
    print("-" * 40)
    print(f"{'Gate':<8} {'Matrix':<25} {'Rotation'}")
    print("-" * 55)
    print(f"{'S':<8} {'[[1,0],[0,i]]':<25} {'90° about Z-axis'}")
    print(f"{'T':<8} {'[[1,0],[0,e^(iπ/4)]]':<25} {'45° about Z-axis'}")
    
    print("\nS gate actions (on equatorial states):")
    for name, state in [('|+⟩', ket_plus)]:
        before = state_to_bloch(state)
        after = state_to_bloch(S @ state)
        print(f"  S{name}: {before} → {tuple(round(x, 2) for x in after)}")
    
    # Gate action summary
    print("\n6. GATE ACTION SUMMARY TABLE")
    print("-" * 60)
    print(f"{'Initial':<10} {'X':<12} {'Y':<12} {'Z':<12} {'H'}")
    print("-" * 60)
    
    states_dict = {
        '|0⟩': ket_0,
        '|1⟩': ket_1,
        '|+⟩': ket_plus,
        '|-⟩': ket_minus
    }
    
    def identify_state(state):
        """Identify a state by comparing to known states."""
        known = {
            (0, 0, 1): '|0⟩',
            (0, 0, -1): '|1⟩',
            (1, 0, 0): '|+⟩',
            (-1, 0, 0): '|-⟩',
            (0, 1, 0): '|+i⟩',
            (0, -1, 0): '|-i⟩'
        }
        coords = tuple(round(x, 1) for x in state_to_bloch(state))
        return known.get(coords, str(coords))
    
    for name, state in states_dict.items():
        x_result = identify_state(X @ state)
        y_result = identify_state(Y @ state)
        z_result = identify_state(Z @ state)
        h_result = identify_state(H @ state)
        print(f"{name:<10} {x_result:<12} {y_result:<12} {z_result:<12} {h_result}")
    
    # Rotation matrices
    print("\n7. GENERAL ROTATION MATRICES")
    print("-" * 40)
    
    def Rx(theta):
        """Rotation about X-axis."""
        return np.array([
            [np.cos(theta/2), -1j*np.sin(theta/2)],
            [-1j*np.sin(theta/2), np.cos(theta/2)]
        ])
    
    def Ry(theta):
        """Rotation about Y-axis."""
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ])
    
    def Rz(theta):
        """Rotation about Z-axis."""
        return np.array([
            [np.exp(-1j*theta/2), 0],
            [0, np.exp(1j*theta/2)]
        ])
    
    print("Rx(θ) = exp(-iθσx/2)")
    print("Ry(θ) = exp(-iθσy/2)")
    print("Rz(θ) = exp(-iθσz/2)")
    print()
    
    # Verify Pauli gates are 180° rotations
    print("Verification: Pauli gates are 180° rotations")
    print(f"  Rx(π) ≈ -iX: {np.allclose(Rx(np.pi), -1j*X)}")
    print(f"  Ry(π) ≈ -iY: {np.allclose(Ry(np.pi), -1j*Y)}")
    print(f"  Rz(π) ≈ -iZ: {np.allclose(Rz(np.pi), -1j*Z)}")
    
    # Composition
    print("\n8. GATE COMPOSITION")
    print("-" * 40)
    print("Sequential gates = sequential rotations")
    print()
    print("Example: H then Z on |0⟩")
    state = ket_0
    print(f"  Start: {state_to_bloch(state)}")
    state = H @ state
    print(f"  After H: {tuple(round(x, 2) for x in state_to_bloch(state))}")
    state = Z @ state
    print(f"  After Z: {tuple(round(x, 2) for x in state_to_bloch(state))}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: Every single-qubit gate is a rotation!")
    print("Understanding rotations = understanding quantum gates.")
    print("=" * 60)

if __name__ == "__main__":
    main()
