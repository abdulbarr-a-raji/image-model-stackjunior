const MIN_WIDTH = 500, MIN_HEIGHT = 500;

async function loadImages() {

    const imageTensors = [];
    const labels = [];
    
    const inputElement = document.getElementById('imageUpload');
    const files = Array.from(inputElement.files);

    for (const file of files) {
        const label = file.name.includes('smiling') ? 1 : 0;

        const img = await readImageFileAsImage(file);
        const tensor = preprocessImage(img);
        
        imageTensors.push(tensor);
        labels.push(label);
    }

    const xs = tf.stack(imageTensors);  // Shape: [batch, height, width, channels]
    const ys = tf.tensor1d(labels, 'int32'); // Shape: [batch]
    document.getElementById('micro-out-div').innerText = labels;

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
    return tf.tidy(() => {
        return tf.browser.fromPixels(img)
        .resizeBilinear([MIN_WIDTH, MIN_HEIGHT])
        .toFloat()
        .div(tf.scalar(255));
    });
}