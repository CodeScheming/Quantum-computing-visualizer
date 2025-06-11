import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit_aer import AerSimulator # For simulating measurements

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Quantum Visualizer", page_icon="⚛️")

# --- Helper Functions ---
def get_statevector_and_probs(qc):
    """Gets statevector and measurement probabilities from a QuantumCircuit."""
    state = Statevector(qc)
    probs = state.probabilities_dict()
    return state, probs

def plot_bloch(statevector):
    """Plots the Bloch sphere for a given statevector."""
    # Qiskit's plot_bloch_multivector expects a Statevector object
    fig = plot_bloch_multivector(statevector)
    return fig

# --- Initialize Session State ---
if 'qc' not in st.session_state:
    st.session_state.qc = QuantumCircuit(1) # Start with a single qubit in |0> state
    st.session_state.history = ["Initial state: |0>"]
    st.session_state.measurement_counts = None

# --- Sidebar for Controls ---
st.sidebar.title("⚛️ Quantum Controls")
st.sidebar.markdown("---")

st.sidebar.header("🔄 Reset Qubit")
if st.sidebar.button("Reset to |0⟩"):
    st.session_state.qc = QuantumCircuit(1)
    st.session_state.history = ["Initial state: |0>"]
    st.session_state.measurement_counts = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header(" GATE OPERATIONS")

gate_options = {
    "Identity (I)": lambda qc, qubit: qc.i(qubit),
    "Pauli-X (NOT)": lambda qc, qubit: qc.x(qubit),
    "Pauli-Y": lambda qc, qubit: qc.y(qubit),
    "Pauli-Z": lambda qc, qubit: qc.z(qubit),
    "Hadamard (H)": lambda qc, qubit: qc.h(qubit),
    "Phase (S)": lambda qc, qubit: qc.s(qubit),
    "S† (Sdg)": lambda qc, qubit: qc.sdg(qubit),
    "T": lambda qc, qubit: qc.t(qubit),
    "T† (Tdg)": lambda qc, qubit: qc.tdg(qubit),
}

selected_gate = st.sidebar.selectbox("Apply Gate:", list(gate_options.keys()))

if st.sidebar.button(f"Apply {selected_gate} Gate"):
    gate_function = gate_options[selected_gate]
    gate_function(st.session_state.qc, 0) # Apply to the 0th qubit
    st.session_state.history.append(f"Applied {selected_gate} gate")
    st.session_state.measurement_counts = None # Clear previous measurement if any
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Rotation Gates")
theta_rx = st.sidebar.slider("Rx Angle (θ)", 0.0, 2 * np.pi, 0.0, step=0.01 * np.pi, format="%.2fπ")
if st.sidebar.button("Apply Rx Gate"):
    st.session_state.qc.rx(theta_rx, 0)
    st.session_state.history.append(f"Applied Rx({theta_rx/np.pi:.2f}π) gate")
    st.session_state.measurement_counts = None
    st.rerun()

theta_ry = st.sidebar.slider("Ry Angle (θ)", 0.0, 2 * np.pi, 0.0, step=0.01 * np.pi, format="%.2fπ")
if st.sidebar.button("Apply Ry Gate"):
    st.session_state.qc.ry(theta_ry, 0)
    st.session_state.history.append(f"Applied Ry({theta_ry/np.pi:.2f}π) gate")
    st.session_state.measurement_counts = None
    st.rerun()

theta_rz = st.sidebar.slider("Rz Angle (φ)", 0.0, 2 * np.pi, 0.0, step=0.01 * np.pi, format="%.2fπ")
if st.sidebar.button("Apply Rz Gate"):
    st.session_state.qc.rz(theta_rz, 0) # Rz is a phase shift, doesn't change |0>, |1> probabilities
    st.session_state.history.append(f"Applied Rz({theta_rz/np.pi:.2f}π) gate")
    st.session_state.measurement_counts = None
    st.rerun()


st.sidebar.markdown("---")
st.sidebar.header("🔬 Measurement")
num_shots = st.sidebar.slider("Number of Shots for Measurement", 1, 10000, 1024)
if st.sidebar.button("Measure Qubit"):
    # Create a new circuit for measurement to not alter the original state for display
    measure_qc = st.session_state.qc.copy()
    measure_qc.measure_all() # Measures all qubits and adds classical bits

    # Use AerSimulator
    simulator = AerSimulator()
    compiled_circuit = transpile(measure_qc, simulator)
    result = simulator.run(compiled_circuit, shots=num_shots).result()
    counts = result.get_counts(measure_qc)
    st.session_state.measurement_counts = counts
    st.session_state.history.append(f"Measured qubit ({num_shots} shots)")
    st.rerun()


# --- Main Page Display ---
st.title("Interactive Quantum Qubit Visualizer")
st.markdown("""
This tool allows you to visualize the state of a single qubit and apply quantum gates.
- **Bloch Sphere:** A geometrical representation of the pure state space of a 1-qubit quantum system.
- **Statevector:** The mathematical representation `α|0⟩ + β|1⟩`, where `|α|² + |β|² = 1`.
- **Probabilities:** The likelihood of measuring the qubit in state `|0⟩` or `|1⟩`.
""")
st.markdown("---")

# Get current state
current_qc = st.session_state.qc
current_statevector, current_probs = get_statevector_and_probs(current_qc)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Bloch Sphere Representation")
    try:
        fig_bloch = plot_bloch(current_statevector)
        st.pyplot(fig_bloch)
        plt.close(fig_bloch) # Close the figure to free memory
    except Exception as e:
        st.error(f"Error plotting Bloch sphere: {e}")

with col2:
    st.subheader("Qubit State Information")

    # Statevector
    st.markdown("**Statevector:**")
    state_arr = current_statevector.data
    alpha_str = f"{state_arr[0].real:.3f}{state_arr[0].imag:+.3f}j"
    beta_str = f"{state_arr[1].real:.3f}{state_arr[1].imag:+.3f}j"
    st.latex(rf"({alpha_str})|0\rangle + ({beta_str})|1\rangle")

    # Probabilities
    st.markdown("**Probabilities:**")
    prob0 = current_probs.get('0', 0.0) # qiskit might represent '0' as '00...0'
    prob1 = current_probs.get('1', 0.0)
    # Ensure keys are present if only one state has non-zero probability after specific gates
    if not current_probs: # e.g. after H then H
        prob0 = np.abs(state_arr[0])**2
        prob1 = np.abs(state_arr[1])**2

    prob_data = {'State': ['|0⟩', '|1⟩'], 'Probability': [prob0, prob1]}
    st.bar_chart(prob_data, x='State', y='Probability', height=200)
    st.text(f"P(|0⟩): {prob0:.3f}")
    st.text(f"P(|1⟩): {prob1:.3f}")

st.markdown("---")
st.subheader("Circuit Diagram")
try:
    # Qiskit's circuit_drawer can be a bit finicky with Matplotlib backends in Streamlit
    # Let's try to draw it and catch errors
    fig_circuit = current_qc.draw(output='mpl', style={'displaycolor': {'h': '#77DDE7'}}) # Example style
    st.pyplot(fig_circuit)
    plt.close(fig_circuit)
except Exception as e:
    st.warning(f"Could not render circuit diagram: {e}. Circuit operations are still applied.")
    st.text(current_qc.draw(output='text'))


st.markdown("---")
st.subheader("Measurement Results")
if st.session_state.measurement_counts:
    st.markdown(f"Results from **{num_shots}** measurement shots:")
    try:
        fig_hist = plot_histogram(st.session_state.measurement_counts)
        st.pyplot(fig_hist)
        plt.close(fig_hist)
    except Exception as e:
        st.error(f"Error plotting histogram: {e}")
else:
    st.info("Perform a measurement to see results here.")


st.markdown("---")
with st.expander("📖 Gate Explanations & History"):
    st.markdown("""
    - **I (Identity):** Does nothing.
    - **X (Pauli-X / NOT):** Flips the qubit state (|0⟩ ↔ |1⟩). Rotation by π around X-axis.
    - **Y (Pauli-Y):** Rotation by π around Y-axis.
    - **Z (Pauli-Z):** Flips the phase of |1⟩. Rotation by π around Z-axis.
    - **H (Hadamard):** Creates superposition. Maps |0⟩ to |+⟩ and |1⟩ to |−⟩.
    - **S (Phase):** Applies a phase of i to |1⟩. Rotation by π/2 around Z-axis.
    - **S† (Sdg):** Conjugate transpose of S. Applies phase of -i to |1⟩.
    - **T:** Rotation by π/4 around Z-axis.
    - **T† (Tdg):** Conjugate transpose of T.
    - **Rx(θ):** Rotation around the X-axis by angle θ.
    - **Ry(θ):** Rotation around the Y-axis by angle θ.
    - **Rz(φ):** Rotation around the Z-axis by angle φ (phase shift).
    """)
    st.subheader("Applied Operations History:")
    for i, item in enumerate(reversed(st.session_state.history)):
        st.text(f"{len(st.session_state.history)-i}. {item}")

st.markdown("---")
st.caption("Built with Streamlit and Qiskit by an AI Assistant")