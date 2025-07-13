# src/model.py
import tensorflow as tf

def classification_model(input_shape, num_classes): # Removed the default value
    """
    Defines the 1D CNN classification model.
    The number of classes is now a required parameter.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    c1 = tf.keras.layers.Conv1D(16, 3, padding="same", activation="relu", kernel_initializer="he_normal")(inputs)
    c1 = tf.keras.layers.Dropout(0.2)(c1)
    
    c2 = tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu", kernel_initializer="he_normal")(c1)
    c2 = tf.keras.layers.Dropout(0.2)(c2)

    c3 = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu", kernel_initializer="he_normal")(c2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)

    f = tf.keras.layers.Flatten()(c3)
    d1 = tf.keras.layers.Dense(128, activation='relu')(f)
    d2 = tf.keras.layers.Dense(32, activation='relu')(d1)

    # The final layer's size is now determined by the num_classes parameter
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(d2)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    print(model.summary())
    return model