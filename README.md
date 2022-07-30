# varroa-detect

Trained classifiers for detecting varroa mites on bees. Each model is available as a quantized TFLite classifier and compiled for the Coral Edge TPU.

<div align="center">
  <img src="varroa-detect/results/pictures/occlusion_sensitivity.png" width="500" height="500">
</div>

## Performance

Prediction scores and inference speeds are documented in ```results/report/report.pdf```.

## Usage

### Edge TPU

The most performant way to make inferences is to use the compiled quantized models with ```edgetpu``` Python API. It can be installed with ```scripts/install_edgetpu_api.sh``` and requires a detected Edge TPU device, like the USB Accelerator or the Dev Board. If a TFLite model which is not fully compiled for the Edge TPU is passed to the ```ClassificationEngine```, uncompiled parts of the computation are executed on the CPU.

Here is an example for how to use the API for single label classification.

```python
import numpy as np
from PIL import Image

from edgetpu.classification.engine import ClassificationEngine


# Load model (Edge TPU is automatically used if possible)
classifier = ClassificationEngine('models/160_160_3_mobilenetv2/model_edgetpu.tflite')

# Input dimensions do not have to match (resize is done by classify_with_image)
input_data = np.array(np.random.random_sample((280, 160, 3)), dtype=np.uint8)

# Accepts PIL image
image = Image.fromarray(input_data, 'RGB')

# Set threshold to smaller than 0 to not omit any predictions
# Predictions are returned as [(label_id, confidence_score)]
prediction[0][1] = classifier.classify_with_image(image, threshold=-1)
```

A detailed description of the API is provided by its [documentation](https://coral.ai/docs/edgetpu/api-intro).

### CPU

In any case, inferences can be done with the TFLite python API. The ```tflite_runtime``` can be installed for the Raspberry Pi 4 with ```scripts/install_tflite_runtime_raspi4```. For other devices refer to the [TFLite guide](https://www.tensorflow.org/lite/guide/python).

Fully quantized TFLite models are found under ```models/<version>/model.tflite```. To perform inferences with these models you can refer to the ```QuantizedClassifier``` class in ```predict_quantized.py```.

## Notes

The code assumes that images are located in ```data/{train|validation|test}/{negative|positive}```.
