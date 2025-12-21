"""
The Bloch Sphere - Concept 4: Physical Interpretation
====================================================
Understanding the geometry of quantum states.
"""

import numpy as np

def main():
    print("=" * 60)
    print("PHYSICAL INTERPRETATION OF THE BLOCH SPHERE")
    print("=" * 60)
    
    # The poles
    print("\n1. THE POLES")
    print("-" * 40)
    print("NORTH POLE: |0⟩")
    print("  θ = 0")
    print("  Measuring always gives outcome 0")
    print("  P(0) = cos²(0/2) = 1")
    print()
    print("SOUTH POLE: |1⟩")
    print("  θ = π")
    print("  Measuring always gives outcome 1")
    print("  P(1) = sin²(π/2) = 1")
    
    # The equator
    print("\n2. THE EQUATOR")
    print("-" * 40)
    print("At θ = π/2, we have equal superpositions:")
    print("  |ψ⟩ = (|0⟩ + e^(iφ)|1⟩)/√2")
    print()
    print("All equatorial states have:")
    print("  • P(0) = P(1) = 50%")
    print("  • Different relative phases (φ)")
    print("  • Different interference patterns")
    
    # Latitude = probability
    print("\n3. LATITUDE = PROBABILITY")
    print("-" * 40)
    print("The polar angle θ directly controls measurement probabilities:")
    print()
    print("  P(|0⟩) = cos²(θ/2)")
    print("  P(|1⟩) = sin²(θ/2)")
    print()
    
    def measurement_probabilities(theta):
        p0 = np.cos(theta / 2) ** 2
        p1 = np.sin(theta / 2) ** 2
        return p0, p1
    
    print(f"{'θ (rad)':<12} {'θ (deg)':<12} {'P(|0⟩)':<12} {'P(|1⟩)'}")
    print("-" * 50)
    
    for theta_deg in [0, 30, 45, 60, 90, 120, 150, 180]:
        theta = np.radians(theta_deg)
        p0, p1 = measurement_probabilities(theta)
        print(f"{theta:.4f}      {theta_deg:3d}°         {p0:.4f}       {p1:.4f}")
    
    # Longitude = phase
    print("\n4. LONGITUDE = PHASE")
    print("-" * 40)
    print("The azimuthal angle φ controls relative phase:")
    print()
    print(f"{'φ':<15} {'State':<15} {'Description'}")
    print("-" * 45)
    print(f"{'0':<15} {'|+⟩':<15} {'Real positive'}")
    print(f"{'π':<15} {'|-⟩':<15} {'Real negative'}")
    print(f"{'π/2':<15} {'|+i⟩':<15} {'Imaginary positive'}")
    print(f"{'3π/2':<15} {'|-i⟩':<15} {'Imaginary negative'}")
    
    # Phase independence of Z-measurement
    print("\n5. PHASE INDEPENDENCE OF Z-MEASUREMENT")
    print("-" * 40)
    print("All equatorial states (θ = π/2) have same Z-measurement probabilities:")
    print()
    
    for phi_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
        phi = np.radians(phi_deg)
        p0, p1 = measurement_probabilities(np.pi/2)
        print(f"φ = {phi_deg:3d}°: P(|0⟩) = {p0:.2f}, P(|1⟩) = {p1:.2f}")
    
    # Antipodal points
    print("\n6. ANTIPODAL POINTS = ORTHOGONAL STATES")
    print("-" * 40)
    print("Points on opposite sides of the sphere are orthogonal:")
    print()
    
    def create_state(theta, phi):
        return np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])
    
    pairs = [
        ((0, 0), (np.pi, 0), '|0⟩ and |1⟩'),
        ((np.pi/2, 0), (np.pi/2, np.pi), '|+⟩ and |-⟩'),
        ((np.pi/2, np.pi/2), (np.pi/2, 3*np.pi/2), '|+i⟩ and |-i⟩')
    ]
    
    for (t1, p1), (t2, p2), desc in pairs:
        s1 = create_state(t1, p1)
        s2 = create_state(t2, p2)
        inner = np.abs(np.vdot(s1, s2))
        print(f"{desc}: |⟨ψ₁|ψ₂⟩| = {inner:.6f}")
    
    # Intuitive summary
    print("\n7. INTUITIVE SUMMARY")
    print("-" * 40)
    print(f"{'Concept':<25} {'Bloch Interpretation'}")
    print("-" * 55)
    print(f"{'Relative phase':<25} {'Longitude (φ)'}")
    print(f"{'Superposition degree':<25} {'Latitude (θ)'}")
    print(f"{'Measurement probability':<25} {'Distance from poles'}")
    print(f"{'Orthogonal states':<25} {'Antipodal points'}")
    print(f"{'Pure vs mixed':<25} {'Surface vs interior'}")
    
    # Simulation
    print("\n8. MEASUREMENT SIMULATION")
    print("-" * 40)
    
    np.random.seed(42)
    
    # State at θ = π/3 (60°)
    theta = np.pi / 3
    state = create_state(theta, 0)
    
    p0_theory = np.cos(theta/2)**2
    p1_theory = np.sin(theta/2)**2
    
    print(f"State: θ = π/3 (60°)")
    print(f"Theoretical: P(|0⟩) = {p0_theory:.4f}, P(|1⟩) = {p1_theory:.4f}")
    print()
    
    # Simulate measurements
    n_shots = 10000
    outcomes = np.random.choice([0, 1], size=n_shots, p=[p0_theory, p1_theory])
    
    p0_exp = np.mean(outcomes == 0)
    p1_exp = np.mean(outcomes == 1)
    
    print(f"Simulated ({n_shots} shots):")
    print(f"  P(|0⟩) = {p0_exp:.4f}")
    print(f"  P(|1⟩) = {p1_exp:.4f}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: The Bloch sphere geometry directly encodes")
    print("the physical properties of quantum states!")
    print("=" * 60)

if __name__ == "__main__":
    main()
