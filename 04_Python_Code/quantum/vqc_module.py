"""
Hybrid Quantum-Classical Learning Component via PennyLane.
Implements:
  1. Classical feature compression to qubit dimension.
  2. Quantum angle encoding using rotation gates R_y and R_z.
  3. Parameterized Variational Quantum Circuit (VQC) with circular CNOT entanglement.
  4. Pauli-Z expectation value measurements.
  5. Exact differentiable quantum-classical backpropagation.
"""
import torch
import torch.nn as nn
import pennylane as qml

class VariationalQuantumCircuit(nn.Module):
    """
    Parameterized Quantum Circuit using PennyLane PyTorch interface.
    """
    def __init__(self, in_features=64, n_qubits=4, n_layers=2, shots=0):
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Compression layer: Classical embedding -> n_qubits angles
        self.compress = nn.Linear(in_features, n_qubits)

        # Setup PennyLane device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        def circuit_fn(inputs, weights):
            # 1. Quantum State Preparation (Angle Encoding with batch broadcasting)
            for i in range(n_qubits):
                qml.RY(inputs[:, i], wires=i)
                qml.RZ(inputs[:, i], wires=i)

            # 2. Parameterized Variational Layers with circular CNOT entanglement
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
                # Entanglement
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

            # 3. Measurement: Pauli-Z expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qnode = qml.QNode(circuit_fn, self.dev, interface="torch", diff_method="backprop")
        self.q_layer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

        # Quantum-classical fusion projection
        self.fusion_gate = nn.Sequential(
            nn.Linear(n_qubits + in_features, in_features),
            nn.GELU()
        )

    def forward(self, classical_embed):
        # classical_embed: [B, in_features]
        # Compress and bound angles to [-pi, pi]
        angles = torch.pi * torch.tanh(self.compress(classical_embed))
        
        # Execute quantum circuit
        q_out = self.q_layer(angles)  # [B, n_qubits]

        # Concatenate and fuse with classical representation via residual projection
        fused = torch.cat([q_out, classical_embed], dim=-1)
        out = self.fusion_gate(fused) + classical_embed
        return out, q_out
