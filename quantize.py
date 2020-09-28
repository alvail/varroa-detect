import tensorflow as tf
import glob


def quantize_model(model_path, input_shape, representative_data_paths):
    """Generate a fully quantized TFLite model from a TensorFlow model."""
    def load_image(filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_png(image, channels=input_shape[2])
        image = tf.image.resize(image, (input_shape[0], input_shape[1]))
        image = tf.expand_dims(image, axis=0)
        return image

    def representative_datagen():
        filenames = representative_data_paths
        for filename in filenames:
            yield [load_image(filename)]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
    # Use 8-bit quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    # Throw error if any operation can not be quantized
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Quantize input and output tensors
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    # Show data to quantize activations
    converter.representative_dataset = representative_datagen
    # Convert and save .tflite
    quantized_model = converter.convert()
    quantized_model_path = model_path + '.tflite'
    with open(quantized_model_path, 'wb') as f:
        f.write(quantized_model)
    return quantized_model_path


if __name__ == '__main__':
    representative_data_paths = glob.glob('data/validation/*/*.png')
    model_paths = glob.glob('models/*/model')
    for model_path in model_paths:
        input_shape = [int(dim) for dim in model_path.split('/')[1].split('_')[:3]]
        quantize_model(model_path, input_shape, representative_data_paths)
