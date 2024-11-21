import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Bước 1: Đường dẫn dữ liệu
data_dir = r'D:\PhanLoaiDongVat\data'

# Bước 2: Xác định các thuộc tính hình ảnh
Image_Width, Image_Height = 224, 224
Image_Size = (Image_Width, Image_Height)
Image_Channels = 3

# Bước 3: Tải MobileNetV2 và thêm các lớp tùy chỉnh
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(Image_Width, Image_Height, Image_Channels))

# Freeze các tầng cơ sở
base_model.trainable = False

# Thêm các tầng head mới
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(9, activation='softmax')  # 9 lớp đầu ra
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Bước 4: Định nghĩa callbacks
earlystop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]

# Bước 5: Chuẩn bị dữ liệu huấn luyện và xác thực
batch_size = 15

train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=15, shear_range=0.1, zoom_range=0.2,
    horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1, validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=Image_Size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=Image_Size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Bước 6: Huấn luyện mô hình
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    steps_per_epoch=train_generator.samples // batch_size,
    callbacks=callbacks
)

# Bước 7: Lưu mô hình
model.save("mobilenetv2_animals.h5")

# Bước 8: Kiểm tra trên ảnh tùy chọn
class_indices = train_generator.class_indices
label_map = {v: k for k, v in class_indices.items()}  # Tạo từ điển để tra cứu nhãn từ chỉ số

def predict_image(model, image_path):
    im = Image.open(image_path)
    im = im.resize(Image_Size)
    im = np.expand_dims(im, axis=0)  # Thêm chiều batch
    im = np.array(im) / 255.0

    pred = np.argmax(model.predict(im), axis=-1)[0]
    result = label_map[pred]
    print(f"Dự đoán: {result}")

# Dự đoán
image_path = r"D:\cho.jpg"
predict_image(model, image_path)
