

import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Quantum State Tomography")

# --- Hilfsfunktionen ---
def generate_density_matrix(bloch_vector):
    x, y, z = bloch_vector
    I = np.eye(2)
    pauli = [np.array([[0, 1], [1, 0]]),  # X
             np.array([[0, -1j], [1j, 0]]),  # Y
             np.array([[1, 0], [0, -1]])]  # Z
    rho = 0.5 * (I + x * pauli[0] + y * pauli[1] + z * pauli[2])
    return rho

def plot_bloch_vector(vec):
    # Kugel Koordinaten
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

    fig = go.Figure()

    # Oberfläche
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        opacity=0.2, colorscale='Blues', showscale=False
    ))

    # Achsen
    for axis, color in zip(['x', 'y', 'z'], ['red', 'green', 'blue']):
        a = dict(x=[-1.2 if axis == 'x' else 0, 1.2 if axis == 'x' else 0],
                 y=[-1.2 if axis == 'y' else 0, 1.2 if axis == 'y' else 0],
                 z=[-1.2 if axis == 'z' else 0, 1.2 if axis == 'z' else 0])
        fig.add_trace(go.Scatter3d(x=a['x'], y=a['y'], z=a['z'], mode='lines', line=dict(color=color, width=4)))

    # Vektor
    fig.add_trace(go.Scatter3d(
        x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
        mode='lines+markers+text',
        line=dict(color='black', width=8),
        marker=dict(size=6, color='black'),
        text=["", "Bloch-Vektor"],
        textposition='top center'
    ))

    fig.update_layout(scene=dict(
        xaxis=dict(title='X', range=[-1.5, 1.5]),
        yaxis=dict(title='Y', range=[-1.5, 1.5]),
        zaxis=dict(title='Z', range=[-1.5, 1.5]),
        aspectmode='cube'
    ), margin=dict(l=0, r=0, t=40, b=0))

    return fig

# --- Session State für Bloch-Vektor ---
if "bloch_vector" not in st.session_state:
    st.session_state.bloch_vector = [0.0, 0.0, 1.0]

def randomize_valid_bloch_vector():
    # Zufälliger Vektor im Einheitsball (r <= 1)
    while True:
        vec = np.random.uniform(-1, 1, 3)
        if np.linalg.norm(vec) <= 1.0:
            st.session_state.bloch_vector = vec
            break

# --- UI: Slider + Randomizer ---
col1, col2 = st.columns([3, 1])
with col1:
    x = st.slider("⟨X⟩", min_value=-1.0, max_value=1.0, value=st.session_state.bloch_vector[0], step=0.01)
    y = st.slider("⟨Y⟩", min_value=-1.0, max_value=1.0, value=st.session_state.bloch_vector[1], step=0.01)
    z = st.slider("⟨Z⟩", min_value=-1.0, max_value=1.0, value=st.session_state.bloch_vector[2], step=0.01)
    bloch_vector = np.array([x, y, z])
    if np.linalg.norm(bloch_vector) <= 1.0:
        st.session_state.bloch_vector = bloch_vector
    else:
        st.warning("Die Länge des Vektors darf maximal 1 sein, sonst ist der Zustand nicht physikalisch!")

with col2:
    if st.button("🔀 Zufälliger Zustand"):
        randomize_valid_bloch_vector()

# --- Dichtematrix anzeigen ---
rho = generate_density_matrix(st.session_state.bloch_vector)
st.markdown("### Dichtematrix ρ")
rho_real = np.round(rho.real, 3)
rho_imag = np.round(rho.imag, 3)
st.latex(r"\rho = " + 
         r"\begin{bmatrix}"
         + f"{rho_real[0,0]} + {rho_imag[0,0]}i & {rho_real[0,1]} + {rho_imag[0,1]}i \\\\"
         + f"{rho_real[1,0]} + {rho_imag[1,0]}i & {rho_real[1,1]} + {rho_imag[1,1]}i"
         + r"\end{bmatrix}"
)

# --- Blochkugel Plot ---
fig = plot_bloch_vector(st.session_state.bloch_vector)
st.plotly_chart(fig, use_container_width=True)
