import glob
import json
from PIL import Image

from edgetpu.classification.engine import ClassificationEngine


if __name__ == '__main__':
    predictions = {
        'path': [],
        'set': [],
        'truth': []
        }

    for split in ['train', 'validation', 'test']:
        for label in ['positive', 'negative']:
            data_paths = glob.glob('data/' + split + '/' + label + '/*.png')
            predictions['path'].extend(data_paths)
            predictions['set'].extend([split] * len(data_paths))
            predictions['truth'].extend([int(label == 'positive')] * len(data_paths))

    inference_times = {}
    model_paths = glob.glob('models/*/model_edgetpu.tflite')

    for model_path in model_paths:
        classifier = ClassificationEngine(model_path)
        model_name = model_path.split('/')[1]
        input_shape = [int(dim) for dim in model_name.split('_')[:3]]
        predictions[model_name] = []
        inference_times[model_name] = []
        for path in predictions['path']:
            image = Image.open(path)
            # Set threshold to smaller than 0 to receive each prediction in range [0, 1]
            prediction = classifier.classify_with_image(image, threshold=-1)
            inference_time = classifier.get_inference_time()
            # Predictions are returned as [(label_id, confidence_score)]
            predictions[model_name].append(prediction[0][1].astype(float))
            inference_times[model_name].append(inference_time)

    with open('results/predictions_edgetpu.json', 'w') as fp:
        json.dump(predictions, fp)

    with open('results/inference_times_edgetpu.json', 'w') as fp:
        json.dump(inference_times, fp)
