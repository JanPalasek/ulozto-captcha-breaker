# ulozto-captcha-breaker
Deep learning model using Tensorflow that breaks ulozto captcha codes. Algorithm used will be described in a standalone document.

## Model specifications
- Input shape: (batch_size, height, width, 1), where height = 70, width = 175
- Output shape: (batch_size, number_of_letters, number_of_classes), where number_of_letters = 4 and number_of_classes = 26

Note that it takes **grayscale images** on the input. RGB images therefore have to be converted.

## How to use pretrained model in your project
### Prerequisities
*numpy* and *tensorflow* package to your virtual machine.

### Steps
1. Go to latest release and download binary files
2. Load model in your project using ```model = tf.keras.models.load_model(PATH_TO_MODEL_DIR)```
  - PATH_TO_MODEL_DIR is path to directory containing the neural network pretrained model
  - it can be found inside the release binary files
3. Copy StringEncoder class from to your project (useful for decoding)
4. Predict using following code
```python
# input of shape (batch_size, 70, 175, 1)
# output will have shape (batch_size, 4, 26)
output = model(input).numpy()
# now get labels
labels_indices = np.argmax(output, axis=2)
decoder = StringEncoder("abcdefghijklmnopqrstuvwxyz")
labels = [decoder.decode(x) for x in labels_indices]
```
- *tf* is alias for tensorflow package, *np* for numpy
