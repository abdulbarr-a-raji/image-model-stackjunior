import { MIN_WIDTH, MIN_HEIGHT } from './data.js';

async function run () {

    const tensors_batch = [];
    for(let ex of ['example1', 'example2', 'example3']) {
        tensors_batch.push(getExample(ex));
    }
    const batch_tensor = tf.stack(tensors_batch);

    const predictor = await loadModel();
    const inferences = makeInferences(predictor, batch_tensor);

    inferences.print();
    console.log("decoded:", await decodePreds(inferences));

}

async function loadModel() {

    const model = await tf.loadLayersModel('http://127.0.0.1:5500//pre-trained model/pretrained-model-v1.json');

    console.log("Model loaded:", model);
    console.log("YAY loaded")

    return model;
    
}

function makeInferences(trainedModel, imgTensors) {

    const output = trainedModel.predictOnBatch(imgTensors);

    return output;

}

function getExample(id) {

    const divElement = document.getElementById(id);
    const divStyles = getComputedStyle(divElement);
    const imageUrl = divStyles.backgroundImage;
    const intWidth = parseInt(divStyles.width.replace("px", ""));
    const intHeight = parseInt(divStyles.height.replace("px", ""));

    const img = new Image();
    img.src = imageUrl.slice(5, -2); // .slice() removes the url("") wrapper text;

    const canvas = document.createElement('canvas');
    canvas.width = intWidth;

    canvas.height = intHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, intWidth, intHeight);

    const imageTensor = tf.browser.fromPixels(canvas)
        .resizeBilinear([MIN_WIDTH, MIN_HEIGHT])
        .toFloat()
        .div(tf.scalar(255));

    return imageTensor;

}

async function decodePreds(preds, legend = {"sad":0, "smiling":1}) {

    const decoded = [];
    const values = await preds.array();
    for(let val of values) {
        for(let [classification, code] of Object.entries(legend)) {
            if (val == code) {
                decoded.push(classification);
                break;
            }
        }
    }

    return decoded;

}

document.getElementById("inference-button").addEventListener("click", run);