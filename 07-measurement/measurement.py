"""
The Bloch Sphere - Concept 7: Measurement
========================================
Projection and state collapse on the Bloch sphere.
"""

import numpy as np

def main():
    print("=" * 60)
    print("QUANTUM MEASUREMENT ON THE BLOCH SPHERE")
    print("=" * 60)
    
    # Define states
    ket_0 = np.array([1, 0])
    ket_1 = np.array([0, 1])
    ket_plus = np.array([1, 1]) / np.sqrt(2)
    ket_minus = np.array([1, -1]) / np.sqrt(2)
    ket_plus_i = np.array([1, 1j]) / np.sqrt(2)
    ket_minus_i = np.array([1, -1j]) / np.sqrt(2)
    
    def state_to_bloch(state):
        """Convert state to Bloch coordinates."""
        rho = np.outer(state, np.conj(state))
        x = 2 * np.real(rho[0, 1])
        y = 2 * np.imag(rho[0, 1])
        z = np.real(rho[0, 0] - rho[1, 1])
        return (x, y, z)
    
    # Measurement as projection
    print("\n1. MEASUREMENT AS PROJECTION")
    print("-" * 40)
    print("Quantum measurement projects the state onto one of the")
    print("measurement basis states.")
    print()
    print("For Z-basis measurement:")
    print("  P(|0⟩) = |⟨0|ψ⟩|² = cos²(θ/2) = (1+z)/2")
    print("  P(|1⟩) = |⟨1|ψ⟩|² = sin²(θ/2) = (1-z)/2")
    
    # Z-basis measurement
    print("\n2. Z-BASIS MEASUREMENT")
    print("-" * 40)
    
    def measure_z(state, num_shots=1):
        """Simulate Z-basis measurement."""
        p0 = np.abs(np.vdot(ket_0, state))**2
        p1 = np.abs(np.vdot(ket_1, state))**2
        outcomes = np.random.choice([0, 1], size=num_shots, p=[p0, p1])
        return outcomes, p0, p1
    
    # Test state at θ = π/3
    theta = np.pi / 3
    test_state = np.array([np.cos(theta/2), np.sin(theta/2)])
    
    print(f"Test state: θ = π/3 (60°)")
    print(f"Bloch coordinates: {tuple(round(x, 3) for x in state_to_bloch(test_state))}")
    print()
    
    _, p0, p1 = measure_z(test_state)
    print(f"Theoretical probabilities:")
    print(f"  P(|0⟩) = cos²(π/6) = {p0:.4f}")
    print(f"  P(|1⟩) = sin²(π/6) = {p1:.4f}")
    
    # X-basis measurement
    print("\n3. X-BASIS MEASUREMENT")
    print("-" * 40)
    print("Measuring in the Hadamard basis {|+⟩, |-⟩}:")
    print()
    print("  P(|+⟩) = |⟨+|ψ⟩|² = (1+x)/2")
    print("  P(|-⟩) = |⟨-|ψ⟩|² = (1-x)/2")
    
    def measure_x(state):
        """Calculate X-basis measurement probabilities."""
        p_plus = np.abs(np.vdot(ket_plus, state))**2
        p_minus = np.abs(np.vdot(ket_minus, state))**2
        return p_plus, p_minus
    
    print(f"\nFor test state (θ = π/3):")
    p_plus, p_minus = measure_x(test_state)
    print(f"  P(|+⟩) = {p_plus:.4f}")
    print(f"  P(|-⟩) = {p_minus:.4f}")
    
    # Y-basis measurement
    print("\n4. Y-BASIS MEASUREMENT")
    print("-" * 40)
    print("Measuring in the circular basis {|+i⟩, |-i⟩}:")
    print()
    print("  P(|+i⟩) = |⟨+i|ψ⟩|² = (1+y)/2")
    print("  P(|-i⟩) = |⟨-i|ψ⟩|² = (1-y)/2")
    
    def measure_y(state):
        """Calculate Y-basis measurement probabilities."""
        p_plus_i = np.abs(np.vdot(ket_plus_i, state))**2
        p_minus_i = np.abs(np.vdot(ket_minus_i, state))**2
        return p_plus_i, p_minus_i
    
    print(f"\nFor test state (θ = π/3):")
    p_plus_i, p_minus_i = measure_y(test_state)
    print(f"  P(|+i⟩) = {p_plus_i:.4f}")
    print(f"  P(|-i⟩) = {p_minus_i:.4f}")
    
    # State collapse
    print("\n5. STATE COLLAPSE")
    print("-" * 40)
    print("After measurement, the state collapses to one of the")
    print("basis states. The original superposition is destroyed!")
    print()
    print("Before measurement: |ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩")
    print("After measuring 0:  |ψ⟩ → |0⟩")
    print("After measuring 1:  |ψ⟩ → |1⟩")
    
    # Simulation
    print("\n6. MEASUREMENT SIMULATION")
    print("-" * 40)
    
    np.random.seed(42)
    n_shots = 10000
    
    print(f"Simulating {n_shots} Z-basis measurements on |+⟩:")
    outcomes, p0, p1 = measure_z(ket_plus, n_shots)
    
    count_0 = np.sum(outcomes == 0)
    count_1 = np.sum(outcomes == 1)
    
    print(f"  Theoretical: P(0) = {p0:.4f}, P(1) = {p1:.4f}")
    print(f"  Observed:    P(0) = {count_0/n_shots:.4f}, P(1) = {count_1/n_shots:.4f}")
    print(f"  Counts:      |0⟩: {count_0}, |1⟩: {count_1}")
    
    # Different states
    print("\n7. MEASUREMENT PROBABILITIES FOR VARIOUS STATES")
    print("-" * 60)
    
    states = [
        ('|0⟩', ket_0),
        ('|1⟩', ket_1),
        ('|+⟩', ket_plus),
        ('|-⟩', ket_minus),
        ('|+i⟩', ket_plus_i),
        ('Custom (θ=π/3)', test_state)
    ]
    
    print(f"{'State':<20} {'P(|0⟩)':<10} {'P(|1⟩)':<10} {'P(|+⟩)':<10} {'P(|-⟩)'}")
    print("-" * 60)
    
    for name, state in states:
        _, p0, p1 = measure_z(state)
        p_plus, p_minus = measure_x(state)
        print(f"{name:<20} {p0:<10.4f} {p1:<10.4f} {p_plus:<10.4f} {p_minus:.4f}")
    
    # Geometric interpretation
    print("\n8. GEOMETRIC INTERPRETATION")
    print("-" * 40)
    print("For any measurement basis:")
    print()
    print("1. Draw a line from the state point perpendicular to")
    print("   the measurement axis")
    print()
    print("2. The projection point determines the probabilities")
    print()
    print("3. After measurement, state jumps to one of the axis")
    print("   endpoints (basis states)")
    
    # Repeated measurement
    print("\n9. REPEATED MEASUREMENT")
    print("-" * 40)
    print("Once collapsed, repeated measurements give same result:")
    print()
    
    # Collapse to |0⟩ and measure again
    collapsed_state = ket_0
    print(f"State after collapse to |0⟩:")
    for i in range(5):
        outcomes, _, _ = measure_z(collapsed_state, 1)
        print(f"  Measurement {i+1}: {outcomes[0]}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: Measurement is irreversible projection.")
    print("The superposition is destroyed upon measurement!")
    print("=" * 60)

if __name__ == "__main__":
    main()
