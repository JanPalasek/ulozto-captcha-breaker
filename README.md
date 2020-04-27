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
3. Normalize image to 0..1 interval. If it already is, skip this step.
```python
img = (img / 255).astype(np.float32)
```
4. Predict using following code
```python
# convert to grayscale
r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
input = 0.299 * r + 0.587 * g + 0.114 * b

# input has now shape (70, 175)
# we modify dimensions to match model's input
input = np.expand_dims(input, 0)
input = np.expand_dims(input, -1)

# input is now of shape (batch_size, 70, 175, 1)
# output will have shape (batch_size, 4, 26)
output = model.predict(input)
# now get labels
labels_indices = np.argmax(output, axis=2)

available_chars = "abcdefghijklmnopqrstuvwxyz"
def decode(li):
    result = []
    for char in li:
        result.append(available_chars[char])
    return "".join(result)

# variable labels will contain read captcha codes
labels = [decode(x) for x in labels_indices]
```
- *tf* is alias for tensorflow package, *np* for numpy
