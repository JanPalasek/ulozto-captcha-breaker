# ulozto-captcha-breaker
Deep learning model using Tensorflow that breaks ulozto captcha codes.

![examples](docs/examples.png)

Algorithm used will be described in a standalone document.

## How to use pretrained model in your project
### Prerequisities
Packages
- *numpy~=1.18.3*
- *tensorflow~=2.1.0*

### Model specification
- Input shape: (batch_size, height, width, 1), where height = 70, width = 175
- Output shape: (batch_size, number_of_letters, number_of_classes), where number_of_letters = 4 and number_of_classes = 26

Note that it takes **grayscale images** on the input. RGB images therefore have to be converted.

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

## How to train your own model
1. Install environment
    Following script creates new virtual environment. You can of course use global environment instead.
    All following section's scripts are expected to be executed from repository's root directory.
    ```shell script
    git clone https://github.com/JanPalasek/ulozto-captcha-breaker
    cd "ulozto-captcha-breaker"
    
    # create virtual environment
    /usr/bin/python3 -m venv "venv"
    
    venv/bin/python3 -m pip install -r "requirements.txt"
    ```
2. Obtain dataset of captcha images and store it to directory *out/data*. Images are expected to be named according
to captcha displayed in the image.

    E.g.
    
    ![captcha image](docs/abfd_ba574f47-92d8-407d-9b34-d5f6fa8a74c3.png)
    
    This captcha image is expected to be named e.g. *ABFD.png*, *abfd.png* (if we don't care about case sensitivity)
    or e.g. *ABFD_{UUID4 CODE}.png* (to distinguish different images for same captcha letters).
    
    This project contains a way to generate captchas yourself using *captcha* Python package using script *src/captcha_generation/simple_captcha_generation.py*.
    You can run it in a following manner
    ```shell script
    venv/bin/python3 src/captcha_generation/simple_captcha_generation.py --height=70 --width=175 --available_chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" --captcha_length=6 --dataset_size=10000
    ```
    
    Some of notable parameters are:
    - *available_chars* - list of characters that will be generated
    - *captcha_length* - how long generated captcha is going to be
    - *dataset_size* - how large dataset is going to be generated
    - *height* - height of generated captcha
    - *width* - width of generated captcha

3. Generate *annotations* files using *src/captcha_annotate.py* script. You can call it for example
    ```shell script
    venv/bin/python3 src/captcha_annotate.py --val_split=0.1 --test_split=0.1
    ```
    This will shuffle and split data into train/validation/test according to following parameters:
    - *val_split* - how large part of data is going to be used for validation, e.g. 0.1 means 10%
    - *test_split* - how large part of data is going to be used for testing
    
    This script will create *annotations.txt*, *annotations-train.txt*, *annotations-validation.txt* and *annotations-test.txt*.

4. Run training script *src/train.py* for example like this:
    ```shell script
    /venv/bin/python3 src/train.py --available_chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" --captcha_length=6 
    ```
   Training script notably logs models after each checkpoint into *logs/train.py-{START TIMESTAMP}-{parameters etc.}* directory.