import pandas as pd
import glob
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from train import get_data_generators


if __name__ == '__main__':
    # Initialize dictionary for saving predictions
    predictions = {
        'path': [],
        'set': [],
        'truth': []
        }
    train_generator, validation_generator, test_generator = get_data_generators(
        data_dir='data',
        target_size=(280, 160),
        batch_size=32,
        augment_train_set=False,
        shuffle=False 
        )

    predictions['set'].extend(['train'] * len(train_generator.filenames))
    predictions['set'].extend(['validation'] * len(validation_generator.filenames))
    predictions['set'].extend(['test'] * len(test_generator.filenames))

    predictions['path'].extend(train_generator.filenames)
    predictions['path'].extend(validation_generator.filenames)
    predictions['path'].extend(test_generator.filenames)

    predictions['truth'].extend(list(train_generator.classes))
    predictions['truth'].extend(list(validation_generator.classes))
    predictions['truth'].extend(list(test_generator.classes))

    model_paths = glob.glob('models/*/model')

    for model_path in model_paths:
        model = keras.models.load_model(model_path)
        model_dir = model_path.split('/')[1]
        input_shape = [int(shape) for shape in model_dir.split('_')[:3]]
        # Save predictions
        train_generator, validation_generator, test_generator = get_data_generators(
            data_dir='data',
            target_size=input_shape[:2],
            batch_size=32,
            augment_train_set=False,
            shuffle=False 
            )
        predictions[model_dir] = []
        predictions[model_dir].extend(list(model.predict(train_generator).ravel()))
        predictions[model_dir].extend(list(model.predict(validation_generator).ravel()))
        predictions[model_dir].extend(list(model.predict(test_generator).ravel()))
        
    # Write predictions to file
    pd.DataFrame.from_dict(predictions).to_csv('results/predictions.csv', sep=',')
