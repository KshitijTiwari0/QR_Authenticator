from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

class HybridModelBuilder:
    def __init__(self, effnet_weights='imagenet'):
        self.effnet_weights = effnet_weights
        
    def build_encoder(self):
        base_model = EfficientNetB0(weights=self.effnet_weights, include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        return Model(inputs=base_model.input, outputs=x)
    
    def build_hybrid(self, effnet_dim, glcm_dim):
        img_input = Input(shape=(effnet_dim,))
        glcm_input = Input(shape=(glcm_dim,))
        
        merged = concatenate([img_input, glcm_input])
        x = Dense(256, activation='relu')(merged)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        out = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=[img_input, glcm_input], outputs=out)
        model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        return model
