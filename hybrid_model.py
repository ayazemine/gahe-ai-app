import tensorflow as tf

class CNNFeatureExtractor:
    """CNN model for feature extraction"""
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = None
        self.feature_extractor = None

    def create_model(self):
        L = tf.keras.layers  # kısaltma

        inputs = L.Input(shape=(self.input_dim, 1))
        x = L.Conv1D(16, kernel_size=2, activation='relu', padding='same')(inputs)
        x = L.MaxPooling1D(pool_size=2, padding='same')(x)
        x = L.Conv1D(64, kernel_size=2, activation='relu', padding='same')(x)
        x = L.MaxPooling1D(pool_size=2, padding='same')(x)
        x = L.Conv1D(256, kernel_size=2, activation='relu', padding='same')(x)
        x = L.MaxPooling1D(pool_size=2, padding='same')(x)
        x = L.Dropout(0.2)(x)

        # Attention + MHA
        attn_out = L.Attention()([x, x])
        mha_out = L.MultiHeadAttention(num_heads=2, key_dim=32)(attn_out, attn_out)
        transformer_out = L.Add()([attn_out, mha_out])
        transformer_out = L.LayerNormalization()(transformer_out)

        features = L.Flatten(name='feature_layer')(transformer_out)
        outputs = L.Dense(1)(features)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit(self, X, y, validation_split=0.2):
        X_reshaped = X.reshape(-1, self.input_dim, 1)
        self.model = self.create_model()
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        self.model.fit(
            X_reshaped, y,
            epochs=500,
            batch_size=16,
            verbose=0,
            validation_split=validation_split,
            callbacks=[early_stop]
        )
        self.feature_extractor = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('feature_layer').output
        )

    def extract_features(self, X):
        X_reshaped = X.reshape(-1, self.input_dim, 1)
        return self.feature_extractor.predict(X_reshaped, verbose=0)