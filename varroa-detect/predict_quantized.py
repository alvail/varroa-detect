import glob
import json
import time
import numpy as np
from PIL import Image

import tflite_runtime.interpreter as tflite


class QuantizedClassifier:
    """Perform inferences with a quantized TFLite model."""
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path)
        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()[0]
        output_details = self.interpreter.get_output_details()[0]

        self.input_tensor_shape = input_details['shape']

        self.input_index = input_details['index']
        self.input_scale = input_details['quantization'][0]
        self.input_zero_point = input_details['quantization'][1]

        self.output_index = output_details['index']
        self.output_scale = output_details['quantization'][0]
        self.output_zero_point = output_details['quantization'][1]
        
    def predict(self, x):
        start_time = time.time()
        # Convert PIL image to numpy array with shape (1, height, width, channels)
        x = np.expand_dims(np.array(x.resize(self.input_tensor_shape[1:3])), 0)
        self.interpreter.set_tensor(self.input_index, np.uint8(x / self.input_scale + self.input_zero_point))
        self.interpreter.invoke()
        prediction = (self.output_scale * (self.interpreter.get_tensor(self.output_index) - self.output_zero_point))
        # Return confidence value and elapsed time
        return (prediction.flatten()[0], (time.time() - start_time) * 1000)


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
    model_paths = glob.glob('models/*/model.tflite')
    
    for model_path in model_paths:
        classifier = QuantizedClassifier(model_path)
        model_name = model_path.split('/')[1]
        input_shape = [int(dim) for dim in model_name.split('_')[:3]]
        predictions[model_name] = []
        inference_times[model_name] = []
        for path in predictions['path']:
            image = Image.open(path)
            prediction, inference_time = classifier.predict(image)
            print(inference_time)
            predictions[model_name].append(prediction.astype(float))
            inference_times[model_name].append(inference_time)

    with open('results/predictions_quantized.json', 'w') as fp:
        json.dump(predictions, fp)

    with open('results/inference_times_quantized.json', 'w') as fp:
        json.dump(inference_times, fp)
        
