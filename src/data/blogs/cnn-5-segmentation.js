export const cnn_5_segmentation = {
  title: "Semantic Segmentation: U-Net থেকে Mask R-CNN — Pixel-Level Prediction",
  description: "FCN, Transposed Convolution, U-Net-এর encoder-decoder architecture ও skip connections, Keras-এ binary segmentation U-Net implementation, Mask R-CNN এবং mIoU metric — pixel-by-pixel classification-এর সম্পূর্ণ গাইড।",
  date: "২৩ মে, ২০২৬",
  category: "কনভোলিউশনাল নিউরাল নেটওয়ার্ক",
  readTime: 12,
  slug: "cnn-semantic-segmentation",
  content: `
    <h3>১. Segmentation-এর বিভিন্ন ধরন</h3>
    <p>Segmentation task-এ প্রতিটি pixel-কে classify করতে হয়। তিনটি প্রধান ধরন:</p>
    <table>
      <thead>
        <tr><th>Type</th><th>Output</th><th>Instance আলাদা?</th><th>Example Application</th></tr>
      </thead>
      <tbody>
        <tr><td>Semantic Segmentation</td><td>Per-pixel class (same class same color)</td><td>না</td><td>Road vs Sky vs Car label</td></tr>
        <tr><td>Instance Segmentation</td><td>Per-object mask + class</td><td>হ্যাঁ</td><td>Car-1 mask, Car-2 mask আলাদা</td></tr>
        <tr><td>Panoptic Segmentation</td><td>Semantic + Instance একসাথে</td><td>হ্যাঁ</td><td>Self-driving full scene understanding</td></tr>
      </tbody>
    </table>
    <p>Medical imaging (tumor detection), autonomous driving (road parsing), satellite imagery (land cover) — এই ক্ষেত্রগুলোয় segmentation অত্যন্ত গুরুত্বপূর্ণ।</p>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt

# Segmentation output format visualization
H, W = 256, 256
num_classes = 3  # Background, Person, Car

# Prediction: (H, W, num_classes) — softmax probabilities
pred_logits = np.random.randn(H, W, num_classes)

# Softmax
exp = np.exp(pred_logits - pred_logits.max(axis=-1, keepdims=True))
pred_probs = exp / exp.sum(axis=-1, keepdims=True)

# Argmax → class map: (H, W)
pred_class_map = np.argmax(pred_probs, axis=-1)

print(f"Prediction probs shape: {pred_probs.shape}")
print(f"Class map shape:        {pred_class_map.shape}")
print(f"Unique classes predicted: {np.unique(pred_class_map)}")

# Per-pixel accuracy
def pixel_accuracy(pred_map, true_map):
    return np.mean(pred_map == true_map)

# Binary segmentation (foreground vs background)
true_binary = np.random.randint(0, 2, (H, W))
pred_binary = np.random.randint(0, 2, (H, W))
print(f"Pixel accuracy: {pixel_accuracy(pred_binary, true_binary):.4f}")
</code></pre>

    <h3>২. FCN: Fully Convolutional Network</h3>
    <p><strong>FCN (Long et al., 2015):</strong> Classification CNN-এর Dense layer-গুলো 1×1 Conv দিয়ে replace করো → output হয় spatial prediction map।</p>
    <p>সমস্যা: MaxPooling-এর কারণে spatial resolution কমে যায়। Solution: bilinear upsampling দিয়ে original resolution restore।</p>
    <p><strong>FCN-32s, FCN-16s, FCN-8s:</strong> কতটা upsample করা হচ্ছে তার উপর নির্ভর করে coarse/fine prediction।</p>
    <pre><code>import tensorflow as tf
from tensorflow.keras import layers, models

def build_fcn(num_classes=21, input_shape=(512, 512, 3)):
    """
    Simplified FCN-like architecture
    Dense layers replace করা হয়েছে Conv2D দিয়ে
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder (VGG-like)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    p1 = layers.MaxPooling2D((2,2))(x)   # /2

    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(p1)
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    p2 = layers.MaxPooling2D((2,2))(x)   # /4

    x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(p2)
    x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
    p3 = layers.MaxPooling2D((2,2))(x)   # /8

    # "Fully connected" layers as 1x1 convolutions
    x = layers.Conv2D(512, (1,1), activation='relu')(p3)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(512, (1,1), activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Score map
    x = layers.Conv2D(num_classes, (1,1))(x)  # (H/8, W/8, num_classes)

    # Upsample 8x back to original size
    outputs = layers.UpSampling2D(size=(8,8), interpolation='bilinear')(x)
    # Final activation
    outputs = layers.Activation('softmax')(outputs)

    return models.Model(inputs, outputs)

fcn = build_fcn(num_classes=21)
print(f"FCN params: {fcn.count_params():,}")
test_input = tf.random.normal((1, 512, 512, 3))
output = fcn(test_input)
print(f"Input:  {test_input.shape}")
print(f"Output: {output.shape}")  # (1, 512, 512, 21)
</code></pre>

    <h3>৩. Transposed Convolution: Learnable Upsampling</h3>
    <p><strong>Transposed Convolution (Deconvolution):</strong> Regular conv-এর বিপরীত — ছোট feature map থেকে বড় output। Learnable weights দিয়ে upsampling।</p>
    <p>Stride=2 transposed conv → output size ≈ 2× input size।</p>
    <p>Bilinear upsampling (fixed) vs Transposed Conv (learnable): সাধারণত transposed conv ভালো result দেয় segmentation-এ।</p>
    <pre><code>import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Bilinear upsampling vs Transposed Conv comparison
x_small = tf.random.normal((1, 8, 8, 64))

# Fixed bilinear upsampling
bilinear_up = layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x_small)

# Learnable transposed convolution
transposed_conv = layers.Conv2DTranspose(
    filters=64,
    kernel_size=(2,2),
    strides=(2,2),
    padding='same'
)(x_small)

print(f"Input shape:         {x_small.shape}")
print(f"Bilinear output:     {bilinear_up.shape}")       # (1, 16, 16, 64)
print(f"TransposedConv out:  {transposed_conv.shape}")   # (1, 16, 16, 64)

# Transposed Conv params
params = 2 * 2 * 64 * 64 + 64  # kernel + bias
print(f"\nTransposed Conv params: {params:,}")
print(f"Bilinear params: 0 (no learning)")

# U-Net-এ Conv2DTranspose + Concatenation pattern
def upsample_block(x, skip_connection, filters):
    """U-Net decoder block"""
    # Upsample
    x = layers.Conv2DTranspose(filters, (2,2), strides=(2,2), padding='same')(x)
    # Concatenate with skip connection from encoder
    x = layers.Concatenate()([x, skip_connection])
    # Two conv layers
    x = layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    return x

print("\nUpsample block created successfully")
</code></pre>

    <h3>৪. U-Net: Encoder-Decoder with Skip Connections</h3>
    <p><strong>U-Net (Ronneberger et al., 2015):</strong> Medical image segmentation-এর জন্য তৈরি, এখন সব ধরনের segmentation-এ ব্যবহার হয়।</p>
    <p>Architecture দুই ভাগ:</p>
    <ul>
      <li><strong>Encoder (Contracting Path):</strong> Conv → Conv → MaxPool — spatial size কমে, channels বাড়ে। Features capture করে।</li>
      <li><strong>Decoder (Expanding Path):</strong> UpConv → Concatenate (encoder skip) → Conv → Conv — spatial size বাড়ে, channels কমে।</li>
    </ul>
    <p><strong>Skip Connections:</strong> Encoder-এর feature map সরাসরি corresponding decoder layer-এ concatenate হয়। Fine-grained spatial information preserve করে।</p>
    <pre><code>import tensorflow as tf
from tensorflow.keras import layers, models

def encoder_block(x, filters, dropout_rate=0.0):
    """U-Net encoder block: Conv → Conv → (Dropout) → MaxPool"""
    x = layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    skip = x                                          # Skip connection save
    x = layers.MaxPooling2D((2,2))(x)
    return x, skip                                    # Return both

def decoder_block(x, skip, filters):
    """U-Net decoder block: TransConv → Concat(skip) → Conv → Conv"""
    x = layers.Conv2DTranspose(filters, (2,2), strides=(2,2), padding='same')(x)
    x = layers.Concatenate()([x, skip])               # Skip connection merge
    x = layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    return x

def build_unet(input_shape=(256, 256, 1), num_classes=1):
    """
    U-Net for binary segmentation (num_classes=1)
    বা multi-class (num_classes > 1)
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder (Contracting Path)
    x, s1 = encoder_block(inputs,  64)              # 256→128, skip1: (256,256,64)
    x, s2 = encoder_block(x,      128)              # 128→64,  skip2: (128,128,128)
    x, s3 = encoder_block(x,      256, dropout_rate=0.3)  # 64→32, skip3
    x, s4 = encoder_block(x,      512, dropout_rate=0.3)  # 32→16, skip4

    # Bottleneck
    x = layers.Conv2D(1024, (3,3), padding='same', activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(1024, (3,3), padding='same', activation='relu')(x)

    # Decoder (Expanding Path)
    x = decoder_block(x, s4, 512)                   # 16→32
    x = decoder_block(x, s3, 256)                   # 32→64
    x = decoder_block(x, s2, 128)                   # 64→128
    x = decoder_block(x, s1,  64)                   # 128→256

    # Output layer
    if num_classes == 1:
        outputs = layers.Conv2D(1, (1,1), activation='sigmoid')(x)
    else:
        outputs = layers.Conv2D(num_classes, (1,1), activation='softmax')(x)

    return models.Model(inputs, outputs)

# Binary segmentation model (1-channel output)
unet = build_unet(input_shape=(256, 256, 1), num_classes=1)
unet.summary()

test = tf.random.normal((2, 256, 256, 1))
out  = unet(test)
print(f"\nU-Net Input:  {test.shape}")
print(f"U-Net Output: {out.shape}")   # (2, 256, 256, 1)
</code></pre>

    <h3>৫. U-Net Training: Binary Segmentation</h3>
    <pre><code>import tensorflow as tf
import numpy as np

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient: 2|A∩B| / (|A|+|B|)"""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    """Combined BCE + Dice loss — segmentation-এ popular"""
    bce  = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice

def iou_score(y_true, y_pred, threshold=0.5, smooth=1e-6):
    """IoU (Jaccard index) metric"""
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)
    y_true_f   = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f   = tf.reshape(y_pred_bin, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# Synthetic training data (real project-এ actual masks দরকার)
def generate_synthetic_data(n=100, img_size=256):
    """Simple synthetic segmentation data: circle in image"""
    images = np.zeros((n, img_size, img_size, 1), dtype=np.float32)
    masks  = np.zeros((n, img_size, img_size, 1), dtype=np.float32)

    for i in range(n):
        # Random circle
        cx = np.random.randint(64, 192)
        cy = np.random.randint(64, 192)
        r  = np.random.randint(20, 60)

        # Image: random noise + circle brighter
        images[i, :, :, 0] = np.random.randn(img_size, img_size) * 0.1
        y_grid, x_grid = np.ogrid[:img_size, :img_size]
        circle_mask = (x_grid - cx)**2 + (y_grid - cy)**2 <= r**2
        images[i, circle_mask, 0] += 1.0
        masks[i, circle_mask, 0]   = 1.0

    return images, masks

x_train, y_train = generate_synthetic_data(200, img_size=128)
x_val,   y_val   = generate_synthetic_data(50,  img_size=128)

# Build smaller U-Net for demo
unet_small = build_unet(input_shape=(128, 128, 1), num_classes=1)
unet_small.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=bce_dice_loss,
    metrics=[dice_coefficient, iou_score]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint('best_unet.h5', save_best_only=True, monitor='val_iou_score', mode='max'),
    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
]

history = unet_small.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=30, batch_size=8,
    callbacks=callbacks, verbose=1
)

# Predict
sample = x_val[:1]
pred   = unet_small.predict(sample)
pred_binary = (pred > 0.5).astype(np.float32)
print(f"Prediction shape: {pred.shape}")
print(f"Max pred value: {pred.max():.4f}")
print(f"Dice score: {dice_coefficient(y_val[:1], pred).numpy():.4f}")
</code></pre>

    <h3>৬. Segmentation Metrics: Pixel Accuracy, IoU ও mIoU</h3>
    <p><strong>Pixel Accuracy:</strong> সঠিকভাবে classified pixel-এর শতাংশ। সমস্যা: class imbalance-এ misleading (90% background → 90% accuracy সহজেই)।</p>
    <p><strong>IoU (Jaccard Index) per class:</strong> IoU_c = |TP_c| / (|TP_c| + |FP_c| + |FN_c|)</p>
    <p><strong>mIoU (mean IoU):</strong> সব class-এর IoU-এর mean। Standard metric।</p>
    <pre><code>import numpy as np

def compute_segmentation_metrics(pred_map, true_map, num_classes):
    """
    pred_map, true_map: (H, W) integer class maps
    Returns: pixel_acc, per_class_iou, mIoU
    """
    # Pixel accuracy
    pixel_acc = np.mean(pred_map == true_map)

    ious = []
    for cls in range(num_classes):
        pred_c = (pred_map == cls)
        true_c = (true_map == cls)

        intersection = np.logical_and(pred_c, true_c).sum()
        union        = np.logical_or(pred_c,  true_c).sum()

        if union == 0:
            iou = float('nan')  # Class not present
        else:
            iou = intersection / union
        ious.append(iou)

    # mIoU: present class-গুলোর mean
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mIoU = np.mean(valid_ious)

    return pixel_acc, ious, mIoU

# Test
H, W = 256, 256
num_classes = 4
true_map = np.random.randint(0, num_classes, (H, W))
pred_map = true_map.copy()
# 20% noise
noise_mask = np.random.rand(H, W) < 0.2
pred_map[noise_mask] = np.random.randint(0, num_classes, noise_mask.sum())

pixel_acc, per_class_iou, mIoU = compute_segmentation_metrics(pred_map, true_map, num_classes)
print(f"Pixel Accuracy: {pixel_acc:.4f}")
print(f"Per-class IoU:  {[f'{iou:.3f}' for iou in per_class_iou]}")
print(f"mIoU:           {mIoU:.4f}")
</code></pre>

    <h3>৭. Mask R-CNN: Instance Segmentation</h3>
    <p><strong>Mask R-CNN (He et al., 2017):</strong> Faster R-CNN-এর উপর একটি segmentation mask branch যোগ করা হয়েছে।</p>
    <p>প্রতিটি detected object-এর জন্য: class label + bounding box + binary mask (28×28)।</p>
    <p><strong>RoI Align:</strong> RoI Pooling-এর improved version — bilinear interpolation দিয়ে misalignment সমাধান।</p>
    <pre><code># Detectron2 দিয়ে Mask R-CNN inference
# pip install detectron2

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

setup_logger()

# Config setup
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Detection threshold
cfg.MODEL.DEVICE = "cpu"  # GPU থাকলে "cuda"

predictor = DefaultPredictor(cfg)

# Image-এ inference
image = cv2.imread("your_image.jpg")
outputs = predictor(image)

# Results
instances = outputs["instances"]
print(f"Detected {len(instances)} instances")
print(f"Classes: {instances.pred_classes}")
print(f"Scores:  {instances.scores}")
print(f"Masks shape: {instances.pred_masks.shape}")  # (N, H, W) boolean

# Visualize
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(instances.to("cpu"))
result_image = out.get_image()[:, :, ::-1]
cv2.imwrite("result.jpg", result_image)
</code></pre>
  `,
};
