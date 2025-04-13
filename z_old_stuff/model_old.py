import os
# Optionally suppress some TF log messages (INFO and WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import glob
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
import time
import logging

# Configure logging for traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== Dummy BLIP Functions =====
def analyze_image_colors(path):
    return ["red", "blue"]

def generate_caption(path):
    return "A sample caption with red and blue tones."

# ===== Data Preparation =====
class MangaMetadataGenerator:
    def __init__(self, data_dir: str, debug_mode: bool = False):
        self.data_dir = data_dir
        self.metadata_cache = {}
        self.debug_mode = debug_mode
        self._preprocess_metadata()

    def _preprocess_metadata(self) -> None:
        logger.info("[1/4] Starting metadata preprocessing...")
        panel_paths = glob.glob(os.path.join(self.data_dir, '**', '*.png'), recursive=True)
        total = len(panel_paths)
        logger.info(f"Found {total} PNG files in {self.data_dir}")
        
        # Check if any images were found
        if total == 0:
            logger.error(f"No PNG files found in {self.data_dir}. Check the path and directory structure.")
            return
            
        processed = 0
        errors = 0
        start_time = time.time()

        # For debug mode, limit to a small number of files
        if self.debug_mode and total > 10:
            logger.info(f"Debug mode: limiting to 10 files instead of {total}")
            panel_paths = panel_paths[:10]
            total = 10

        for idx, path in enumerate(panel_paths):
            if idx % 10 == 0:
                elapsed = time.time() - start_time
                logger.info(f"[1/4] Processed {idx}/{total} panels | Errors: {errors} | Time: {elapsed:.1f}s")
            metadata_path = path.replace('.png', '.json')
            try:
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                    self.metadata_cache[path] = metadata_path
                    processed += 1
                    continue

                logger.info(f"[1/4] Analyzing {os.path.basename(path)}...")
                colors = analyze_image_colors(path)
                caption = generate_caption(path)
                metadata = {
                    "colors": [{"name": c} for c in colors],
                    "caption": caption,
                    "processed_time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                self.metadata_cache[path] = metadata_path
                processed += 1
            except Exception as e:
                errors += 1
                logger.error(f"[1/4] Error processing {path}: {str(e)}")
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
        
        logger.info(f"[1/4] Completed: {processed} processed, {errors} errors, {len(self.metadata_cache)} valid entries")
        
        # Check if any valid entries were found
        if len(self.metadata_cache) == 0:
            logger.error("No valid panel metadata pairs were found. Cannot proceed with training.")

    def _metadata_to_vector(self, metadata: dict) -> np.ndarray:
        try:
            color_features = np.zeros(8, dtype=np.float32)
            color_categories = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
            for color in metadata.get("colors", []):
                color_name = color.get("name", "").lower()
                for idx, cat in enumerate(color_categories):
                    if cat in color_name:
                        color_features[idx] = 1.0
            caption_len = min(len(metadata.get("caption", "")) / 100.0, 1.0)
            return np.concatenate([color_features, np.array([caption_len], dtype=np.float32)])
        except Exception as e:
            logger.error(f"Metadata conversion error: {str(e)}")
            return np.zeros(9, dtype=np.float32)

    def composite_generator(self, batch_size: int = 4, panel_shape: tuple = (720, 1280), grid: tuple = (1, 1), max_batches: int = None):
        """
        Note: panel_shape is (width, height) for PIL.resize.
        PIL.resize(panel_shape) returns an array with shape (height, width, channels).
        Here, using (720,1280) yields arrays of shape (1280,720,3).

        Yields incomplete batches instead of stalling if there aren't enough panels.
        Added max_batches parameter to limit iterations for debugging.
        """
        logger.info("[2/4] Initializing composite generator...")
        valid_paths = [p for p in self.metadata_cache.keys() if os.path.exists(p)]
        
        # Check if we have valid paths
        total_panels = len(valid_paths)
        if total_panels == 0:
            logger.error("[2/4] No valid panel paths available for generator. Check metadata preprocessing.")
            return
            
        panels_per_sample = grid[0] * grid[1]  # 1 panel per sample
        logger.info(f"[2/4] Total panels: {total_panels} | Panels per sample: {panels_per_sample} | Batch size: {batch_size}")
        
        # Debug mode: use smaller batches and limit panels
        if self.debug_mode:
            batch_size = min(batch_size, 2)
            logger.info(f"[2/4] Debug mode: using batch size {batch_size}")
            if len(valid_paths) > 10:
                valid_paths = valid_paths[:10]
                logger.info(f"[2/4] Debug mode: limiting to 10 panels")
        
        batch_num = 0
        try:
            while True:
                # Check if we've reached max_batches (for debugging)
                if max_batches is not None and batch_num >= max_batches:
                    logger.info(f"[2/4] Reached max_batches limit of {max_batches}")
                    break
                    
                np.random.shuffle(valid_paths)
                logger.info(f"[2/4] Starting epoch shuffle | Remaining panels: {len(valid_paths)}")
                
                for i in range(0, len(valid_paths), panels_per_sample * batch_size):
                    batch_paths = valid_paths[i:i + panels_per_sample * batch_size]
                    if len(batch_paths) < panels_per_sample * batch_size:
                        logger.info(f"[2/4] Incomplete batch with {len(batch_paths)} panels, yielding anyway.")
                    
                    batch_panels = []
                    batch_conditions = []
                    start_time = time.time()
                    
                    # Process each panel in the batch
                    for j, path in enumerate(batch_paths):
                        try:
                            logger.debug(f"[2/4] Processing panel {j+1}/{len(batch_paths)}: {os.path.basename(path)}")
                            
                            # Check image file
                            if not os.path.exists(path):
                                logger.error(f"[2/4] Image file not found: {path}")
                                continue
                                
                            # Open and resize image
                            img = Image.open(path).convert("RGB").resize(panel_shape)
                            panel = np.array(img, dtype=np.float32) / 255.0
                            
                            # Check metadata file
                            if path not in self.metadata_cache or not os.path.exists(self.metadata_cache[path]):
                                logger.error(f"[2/4] Metadata file not found for: {path}")
                                continue
                                
                            # Load metadata and convert to condition vector
                            with open(self.metadata_cache[path], 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            condition = self._metadata_to_vector(metadata)
                            
                            # Add to batch
                            batch_panels.append(panel)
                            batch_conditions.append(condition)
                            
                        except Exception as e:
                            logger.error(f"[2/4] Failed processing {path}: {str(e)}")
                            continue
                    
                    # Yield batch if not empty
                    if len(batch_panels) > 0:
                        elapsed = time.time() - start_time
                        logger.info(f"[2/4] Yielded batch {batch_num} | Panel shape: {batch_panels[0].shape if batch_panels else 'N/A'} | Time: {elapsed:.1f}s")
                        batch_num += 1
                        yield [np.array(batch_panels), np.array(batch_conditions)], np.array(batch_panels)
                    else:
                        logger.warning("[2/4] Empty batch, nothing to yield.")
                        
                    # For debug mode, limit to one batch
                    if self.debug_mode and batch_num >= 1:
                        logger.info("[2/4] Debug mode: stopping after first batch")
                        return
        
        except Exception as e:
            logger.error(f"[2/4] Generator exception: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

# ===== U-Net Inspired Conditional Autoencoder with Condition Injection =====
def build_unet_conditional_autoencoder(input_shape: tuple, condition_dim: int) -> Model:
    """
    Build a U-Net style conditional autoencoder.
    The encoder downsamples the image and the bottleneck has shape (160,90,256)
    for input (1280,720,3). Instead of flattening the bottleneck (which is huge),
    we process the condition vector into a feature map of shape (160,90,1) and
    concatenate it with the bottleneck.
    """
    logger.info("[3/4] Building U-Net inspired autoencoder architecture...")
    # Encoder
    img_input = Input(shape=input_shape, name='img_input')
    
    # Block 1
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)  # (640,360,32)
    
    # Block 2
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)  # (320,180,64)
    
    # Block 3
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)  # (160,90,128)
    
    # Bottleneck
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)  # (160,90,256)
    
    # Process condition vector and inject into bottleneck
    condition_input = Input(shape=(condition_dim,), name='condition_input')
    cond_dense = layers.Dense(128, activation='relu', name='cond_dense')(condition_input)
    # Project condition to match the spatial dimensions of c4: (160*90 = 14400)
    cond_map = layers.Dense(160 * 90, activation='relu')(cond_dense)
    cond_map = layers.Reshape((160, 90, 1))(cond_map)
    
    # Concatenate condition map with bottleneck features along channels.
    combined = layers.Concatenate(axis=-1, name='combined_features')([c4, cond_map])  # shape: (160,90,257)
    # Use a 1x1 convolution to merge the channels back to 256.
    c4_modified = layers.Conv2D(256, (1, 1), activation='relu', padding='same', name='latent_conv')(combined)
    
    # Decoder with skip connections
    u1 = layers.UpSampling2D((2,2))(c4_modified)  # (320,180,256)
    u1 = layers.Concatenate()([u1, c3])             # (320,180,256+128)
    c5 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(u1)
    c5 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c5)
    
    u2 = layers.UpSampling2D((2,2))(c5)             # (640,360,128)
    u2 = layers.Concatenate()([u2, c2])             # (640,360,128+64)
    c6 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(u2)
    c6 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c6)
    
    u3 = layers.UpSampling2D((2,2))(c6)             # (1280,720,64)
    u3 = layers.Concatenate()([u3, c1])             # (1280,720,64+32)
    c7 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(u3)
    c7 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(c7)
    
    decoded = layers.Conv2D(3, (1,1), activation='sigmoid', name='final_output')(c7)
    logger.info(f"Model output shape: {decoded.shape}")
    
    return Model(inputs=[img_input, condition_input], outputs=decoded)

# ===== Inference: Build Generator from Trained Autoencoder =====
def build_generator_from_autoencoder(autoencoder: Model, condition_dim: int) -> Model:
    """
    Build a generator that takes only the condition input.
    Here, we bypass the image encoder by feeding a dummy zero image.
    (Skip connections during inference are more complex; this is a simplified version.)
    """
    condition_input = Input(shape=(condition_dim,), name='gen_condition_input')
    # Process condition through the same branch.
    cond_dense = autoencoder.get_layer('cond_dense')(condition_input)
    cond_map = layers.Dense(160 * 90, activation='relu')(cond_dense)
    cond_map = layers.Reshape((160, 90, 1))(cond_map)
    # Create a dummy bottleneck of zeros with shape (160,90,256)
    dummy_bottleneck = layers.Lambda(lambda x: tf.zeros((tf.shape(x)[0], 160, 90, 256)), name='dummy_bottleneck')(cond_map)
    # Concatenate condition map with dummy bottleneck.
    combined = layers.Concatenate(axis=-1)([dummy_bottleneck, cond_map])
    # Pass through the same 1x1 conv to produce a latent feature map.
    c4_modified = autoencoder.get_layer('latent_conv')(combined)
    
    # Now use a simplified decoder path (for inference, skip connections are approximated)
    decoder_layers = [
        layers.UpSampling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.UpSampling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.UpSampling2D((2,2)),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.Conv2D(32, (3,3), activation='relu', padding='same')
    ]
    
    x = c4_modified
    for layer in decoder_layers:
        x = layer(x)
    
    decoded = layers.Conv2D(3, (1,1), activation='sigmoid', name='generator_output')(x)
    generator = Model(inputs=condition_input, outputs=decoded, name='generator')
    return generator

# ===== Helper: Convert Text Prompt to Condition Vector =====
def text_to_condition(text: str) -> np.ndarray:
    color_categories = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
    condition = np.zeros(9, dtype=np.float32)
    text_lower = text.lower()
    for idx, color in enumerate(color_categories):
        if color in text_lower:
            condition[idx] = 1.0
    caption_len = min(len(text) / 100.0, 1.0)
    condition[8] = caption_len
    return condition

# ===== Inference Pipeline =====
def test_inference(autoencoder: Model, text_prompt: str, save_path: str = "generated_webtoon_panel.png"):
    CONDITION_DIM = 9
    logger.info(f"[Inference] Building generator from autoencoder...")
    generator = build_generator_from_autoencoder(autoencoder, CONDITION_DIM)
    logger.info(f"[Inference] Converting text prompt to condition vector...")
    condition = text_to_condition(text_prompt)
    condition = np.expand_dims(condition, axis=0)
    logger.info(f"[Inference] Generating image from prompt: '{text_prompt}'")
    generated_image = generator.predict(condition)
    generated_image = (generated_image[0] * 255).astype(np.uint8)
    Image.fromarray(generated_image).save(save_path)
    logger.info(f"[Inference] Generated webtoon panel saved to {save_path}")

# ===== Training Pipeline with Debug Mode =====
def train(debug_mode=False):
    logger.info("[0/4] Initializing training pipeline...")
    DATA_DIR = "manga_chapters"  # Folder with your webtoon panels
    
    # Our model input is (height, width, channels) = (1280,720,3)
    INPUT_SHAPE = (1280, 720, 3)
    CONDITION_DIM = 9
    
    # Adjust batch size and epochs for debug mode
    BATCH_SIZE = 1 if debug_mode else 2
    EPOCHS = 1 if debug_mode else 100
    STEPS_PER_EPOCH = 2 if debug_mode else 5

    logger.info("[0/4] Initializing metadata generator...")
    generator_data = MangaMetadataGenerator(DATA_DIR, debug_mode=debug_mode)
    
    # Check if we have valid panels
    if len(generator_data.metadata_cache) == 0:
        logger.error("No valid panels found. Please check your data directory and ensure it contains PNG files.")
        return None
        
    # Check if directory exists and is not empty
    if not os.path.exists(DATA_DIR):
        logger.error(f"Data directory {DATA_DIR} does not exist. Create it or specify correct path.")
        return None
        
    if not os.listdir(DATA_DIR):
        logger.error(f"Data directory {DATA_DIR} is empty. Add manga panel images.")
        return None
        
    # Limited number of batches for debug mode
    max_batches = 3 if debug_mode else None
    train_gen = generator_data.composite_generator(
        batch_size=BATCH_SIZE, 
        panel_shape=(720, 1280),
        max_batches=max_batches
    )

    logger.info("[3/4] Building model...")
    autoencoder = build_unet_conditional_autoencoder(INPUT_SHAPE, CONDITION_DIM)
    autoencoder.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')

    logger.info("[4/4] Starting training...")
    try:
        autoencoder.fit(
            train_gen,
            steps_per_epoch=STEPS_PER_EPOCH,
            epochs=EPOCHS,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint('blip_conditional_ae.keras', save_best_only=True, verbose=1),
                tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1),
                tf.keras.callbacks.CSVLogger('training_log.csv')
            ],
            verbose=2
        )
        logger.info("[4/4] Training completed!")
        autoencoder.save('blip_conditional_ae.keras')
        return autoencoder
    except Exception as e:
        logger.error(f"[4/4] Training error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# ===== Main with Debug Mode =====
if __name__ == '__main__':
    # Run in debug mode first
    debug_mode = False
    logger.info(f"Running in {'DEBUG' if debug_mode else 'NORMAL'} mode")
    
    try:
        autoencoder_model = train(debug_mode=debug_mode)
        
        if autoencoder_model is not None:
            # Test with a simple prompt
            prompt = "A dramatic scene with red and blue highlights."
            test_inference(autoencoder_model, prompt, save_path="generated_webtoon_panel.png")
        else:
            logger.error("Training failed or was interrupted. No model available for inference.")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())