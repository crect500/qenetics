from keras import backend, Layer, layers, models, regularizers

class Model(object):
    """Abstract model call.

    Abstract class of the DNA model.

    Args
    ----
    dropout: The dropout rate.
    layer1_decay: The first layer's weight decay.
    layer2_decay: The second layer's weight decay.
    init_name: The name of the Keras initialization method.
    """

    __slots__ = (
        "dropout",
        "layer1_decay",
        "layer2_decay",
        "init_name",
        "model_name",
        "scope",
    )

    def __init__(
        self,
        dropout=0.0,
        layer1_decay=0.0,
        layer2_decay=0.0,
        init_name="glorot_uniform",
    ):
        self.dropout = dropout
        self.layer1_decay = layer1_decay
        self.layer2_decay = layer2_decay
        self.init_name = init_name
        self.model_name = self.__class__.__name__
        self.scope = None

    def inputs(self, *args, **kwargs):
        """Return list of Keras model inputs."""
        pass

    def _build(self, input, output):
        """Build final model at the end of `__call__`."""
        model = models.Model(input, output, name=self.model_name)
        if self.scope:
            for layer in model.layers:
                if layer not in model.input_layers:
                    layer.name = f"{self.scope}/{layer.name}"
        return model

    def __call__(self, inputs=None):
        """Build the model.

        Args
        ----
        inputs: Keras model inputs
        """
        pass


class DnaModel(Model):
    """Abstract class of a DNA model."""

    def __init__(self, *args, **kwargs):
        super(DnaModel, self).__init__(*args, **kwargs)
        self.scope = "dna"

    def inputs(self, dna_wlen: int):
        return [layers.Input(shape=(dna_wlen, 4), name="dna")]


class CnnL2h128(DnaModel):
    """CNN with two convolutional and one fully-connected layer with 128 units.

    Parameters: 4,100,000
    Specification: conv[128@11]_mp[4]_conv[256@3]_mp[2]_fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL2h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = regularizers.WeightRegularizer(
            l1=self.layer1_decay, l2=self.layer2_decay
        )
        x = layers.Conv1D(128, 11, init=self.init_name, W_regularizer=w_reg)(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling1D(4)(x)

        w_reg = regularizers.WeightRegularizer(
            l1=self.layer1_decay, l2=self.layer2_decay
        )
        x = layers.Conv1D(256, 3, init=self.init_name, W_regularizer=w_reg)(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling1D(2)(x)

        x = layers.Flatten()(x)

        w_reg = regularizers.WeightRegularizer(
            l1=self.layer1_decay, l2=self.layer2_decay
        )
        x = layers.Dense(
            self.nb_hidden, init=self.init_name, W_regularizer=w_reg
        )(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class ScaledSigmoid(layers.Layer):
    """Scaled sigmoid activation function.

    Scales the maximum of the sigmoid function from one to the provided value.

    Parameters
    ----------
    scaling: float
        Maximum of sigmoid function.
    """

    def __init__(self, scaling=1.0, **kwargs):
        self.supports_masking = True
        self.scaling = scaling
        super(ScaledSigmoid, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return backend.sigmoid(x) * self.scaling

    def get_config(self):
        config = {"scaling": self.scaling}
        base_config = super(ScaledSigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def add_output_layers(model: Model) -> Layer:
    """
    Add and return outputs to a given layer.

    Adds output layer for each output in `output_names` to `model`.

    Args
    ----
    model: Keras model to which output layers are added.
    output_names: List of output names correlating to desired output metrics.

    Returns
    -------
    Output layers added to `model`.
    """
    return layers.Dense(
        1, init="glorot_uniform", activation="sigmoid", name="output"
    )(model)


def _get_first_conv_layer(
    model_layers: list[Layer], get_activation=False
) -> Layer | tuple[Layer, Layer]:
    """Return the first convolutional layers in a stack of layer.

    Args
    ----
    model_layers: List of Keras layers.
    get_activation: Return the activation layer after the convolutional weight layer.

    Returns
    -------
    Convolutional layer or tuple of convolutional layer and activation layer if
    `get_activation=True`.
    """
    convulational_layer: Layer | None = None
    activation_layer: Layer | None = None
    for layer in model_layers:
        if isinstance(layer, layers.Conv1D) and layer.input_shape[-1] == 4:
            convulational_layer = layer
            if not get_activation:
                break
        elif convulational_layer and isinstance(layer, layers.Activation):
            activation_layer = layer
            break
    if not convulational_layer:
        raise ValueError("Convolutional layer not found")
    if get_activation:
        if not activation_layer:
            raise ValueError("Activation layer not found")
        return convulational_layer, activation_layer
    else:
        return convulational_layer
