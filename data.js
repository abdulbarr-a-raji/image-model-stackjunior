const MIN_WIDTH = 512, MIN_HEIGHT = 512;

async function loadImages() {

    const imageTensors = [];
    const labels = [];
    
    const inputElement = document.getElementById('imagedataUpload');
    const files = Array.from(inputElement.files);

    for (const file of files) {
        const label = file.name.includes('smiling') ? 1 : 0;

        const img = await readImageFileAsImage(file);
        const tensor = preprocessImage(img);
        // console.log(tensor);
        
        imageTensors.push(tensor);
        labels.push(label);
    }

    /* must catch this error:
        Uncaught (in promise) Error: 
        Pass at least one tensor to tf.stack
        at loadImages (data.js)

        | it occurs when no images are uploaded, 
        | but button has been pressed
     */
    // console.log(imageTensors);
    console.log(imageTensors.length);

    const xs = tf.stack(imageTensors);  // tensor shape: [batch, height, width, channels]
    const ys = tf.tensor1d(labels, 'int32'); // tensor shape: [batch]
    document.getElementById('micro-out-div').innerText = labels;

    // console.log(xs);
    // console.log();
    // console.log(ys);

    return [xs, ys];

}

function readImageFileAsImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = () => {
            const img = new Image();
            img.src = reader.result;

            img.onload = () => resolve(img);
            img.onerror = reject;
        };

        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

function preprocessImage(img) {

    // console.log(img.naturalWidth+"x"+img.naturalHeight);

    return tf.tidy(() => {
        return tf.browser.fromPixels(img)
        .resizeBilinear([MIN_WIDTH, MIN_HEIGHT])
        .toFloat()
        .div(tf.scalar(255));
    });

}

export {loadImages};