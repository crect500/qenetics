import pennylane as qp

DEFAULT_DEVICE: str = "default.qubit"
DEVICES: tuple[str, ...] = (
    DEFAULT_DEVICE,
    "lightning.qubit",
    "lightning.kokkos",
    "lightning.gpu",
)


def device_setup(
    device_name: str, wire_quantity: int, distribute: bool = False
) -> qp.devices.Device:
    """
    Create a PennyLane device.

    Args
    ----
    device_name: One of `DEVICES`. Lightning devices need their plugins.
    wire_quantity: The number of wires.
    distribute: For lightning.gpu, batch observables across GPUs.

    Returns
    -------
    The device.

    Raises
    ------
    ValueError if the device name is not supported.
    """
    if device_name not in DEVICES:
        raise ValueError(
            f"Device name {device_name} not recognized; expected one of "
            f"{', '.join(DEVICES)}"
        )

    if device_name == "lightning.gpu":
        return qp.device(device_name, wires=wire_quantity, batch_obs=distribute)

    return qp.device(device_name, wires=wire_quantity)
