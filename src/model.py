import tensorflow as tf
from keras import Model, layers


class DepthwiseSeparableResBlock(layers.Layer):
    """
    Residual block utilizing Depthwise Separable Convolutions.
    Automatically applies a 1x1 projection shortcut if spatial or channel
    dimensions change.
    """

    def __init__(self, filters, stride=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self.activation_name = activation
        self.activation_fn = layers.Activation(activation)

        # Main path: First separable conv
        self.sep_conv1 = layers.SeparableConv2D(
            filters=filters,
            kernel_size=3,
            strides=stride,
            padding="same",
            use_bias=False,
        )
        self.bn1 = layers.BatchNormalization()

        # Main path: Second separable conv
        self.sep_conv2 = layers.SeparableConv2D(
            filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False
        )
        self.bn2 = layers.BatchNormalization()

        self.add = layers.Add()
        self.shortcut = None

    def build(self, input_shape):
        in_channels = input_shape[-1]

        # Apply 1x1 projection if spatial dimensions shrink or channels mismatch
        if self.stride != 1 or in_channels != self.filters:
            self.shortcut = tf.keras.Sequential(
                [
                    layers.Conv2D(
                        filters=self.filters,
                        kernel_size=1,
                        strides=self.stride,
                        padding="same",
                        use_bias=False,
                    ),
                    layers.BatchNormalization(),
                ]
            )
        else:
            self.shortcut = layers.Identity()

        super().build(input_shape)

    def call(self, inputs, training=False):
        # Shortcut branch
        residual = self.shortcut(inputs, training=training)

        # Feature processing branch
        x = self.sep_conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation_fn(x)

        x = self.sep_conv2(x)
        x = self.bn2(x, training=training)

        # Merge and activate
        x = self.add([x, residual])
        return self.activation_fn(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "stride": self.stride,
                "activation": self.activation_name,
            }
        )
        return config


def build_separable_resnet(input_shape=(256, 256, 1)):
    inputs = layers.Input(shape=input_shape)

    # Stem: Standard conv to build dense initial feature maps
    x = layers.Conv2D(8, kernel_size=3, strides=3, padding="same", use_bias=False)(
        inputs
    )
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="valid")(x)

    # Stage 1: Preserve dimensions (stride=1)
    x = DepthwiseSeparableResBlock(filters=16, stride=1)(x)
    # x = DepthwiseSeparableResBlock(filters=8, stride=1)(x)

    # Stage 2: Downsample spatial grid, double channels
    x = DepthwiseSeparableResBlock(filters=32, stride=1)(x)
    # x = DepthwiseSeparableResBlock(filters=16, stride=1)(x)

    # # Stage 3: Downsample spatial grid, double channels
    # x = DepthwiseSeparableResBlock(filters=32, stride=2)(x)
    # x = DepthwiseSeparableResBlock(filters=32, stride=1)(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs, name="separable_resnet")
    return model
