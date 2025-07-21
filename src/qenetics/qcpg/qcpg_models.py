import jax
import pennylane as qml


def strongly_entangled_jax(parameters: jax.Array) -> None:
    layer_quantity: int = parameters.shape[0]
    wire_quantity: int = parameters.shape[1]
    rotations_quantity: int = parameters.shape[2]
    for layer_index in range(layer_quantity):
        qml.StronglyEntanglingLayers(
            parameters[layer_index, :, :].reshape(
                (1, wire_quantity, rotations_quantity)
            ),
            range(wire_quantity),
        )
