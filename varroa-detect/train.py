import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_data_generators(data_dir, target_size, batch_size, augment_train_set=True, shuffle=True):
    """Create data generators with optional augmentation for the training set."""
    datagen = ImageDataGenerator()
    if augment_train_set == True:
        train_datagen = ImageDataGenerator(
            rotation_range=90,
            shear_range=30,
            fill_mode='nearest',
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=(1, 1.3),
            brightness_range=(0.7, 1.3)
            )
    else:
        train_datagen = datagen
    train_generator = train_datagen.flow_from_directory(
        data_dir + '/train',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=shuffle,
        seed=42
        )
    validation_generator = datagen.flow_from_directory(
        data_dir + '/validation',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=shuffle,
        seed=42
        )
    test_generator = datagen.flow_from_directory(
        data_dir + '/test',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=shuffle,
        seed=42
        )
    return train_generator, validation_generator, test_generator


def prepare_model(input_shape, learning_rate, architecture='mobilenetv2'):
    """Prepare CNN classifier for training."""
    if architecture == 'mobilenetv2':
        # Load MobileNetV2 trained on ImageNet for transfer learning
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
            )
        # Unlock all weights for retraining
        base_model.trainable = True
        # Add bottom and top layers
        model = keras.Sequential([
            # Rescaling layer for faster preprocessing
            preprocessing.Rescaling(1.0 / 255),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1, activation='sigmoid')
            ])
    else:
         raise ValueError('chosen architecture is not available')
    # Specify optimizer and loss function
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
            ]
        )  
    return model


def train_model(model, train_data, validation_data, epochs, checkpoint_path):
    """Train model and checkpoint epoch with highest validation accuracy."""
    callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
        )
    model.fit(
        train_data,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=[callback]
        )


if __name__ == '__main__':
    for architecture in ['mobilenetv2']:
        for input_shape in [(96, 96, 3), (128, 128, 3), (160, 160, 3)]:
            # Train model
            train_generator, validation_generator, test_generator = get_data_generators(
                data_dir='data',
                target_size=input_shape[:2],
                batch_size=32,
                augment_train_set=True,
                shuffle=True 
                )
            model = prepare_model(
                input_shape=input_shape,
                learning_rate=0.001,
                architecture=architecture
                )
            model_dir = '{}_{}_{}_{}'.format(input_shape[0], input_shape[1], input_shape[2], architecture)
            checkpoint_path = 'models/' + model_dir + '/model'
            train_model(
                model=model,
                train_data=train_generator,
                validation_data=validation_generator,
                epochs=50,
                checkpoint_path=checkpoint_path
                )
