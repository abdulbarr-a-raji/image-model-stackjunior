// import * as tf from '@tensorflow/tfjs';

function run () {

    const predictor = loadModel();
    const ex1_inference = makeInference(predictor, getExample("example1"));
    const ex2_inference = makeInference(predictor, getExample("example2"));
    const ex3_inference = makeInference(predictor, getExample("example3"));

    console.log("1st example:", ex1_inference);
    console.log("2nd example:", ex2_inference);
    console.log("3rd example:", ex3_inference);

}

async function loadModel() {

    const model = await tf.loadLayersModel('http://127.0.0.1:5500//pre-trained model/pretrained-model-v1.json');

    console.log("Model loaded:", model);
    console.log("YAY loaded")

    return model;
    
}

function makeInference(trainedModel, imgTensor) {

    const output = trainedModel.predict(imgTensor);

    return output;

}

function getExample(id) {

    const divElement = document.getElementById(id);
    const divStyles = getComputedStyle(divElement);
    const imageUrl = divStyles.backgroundImage;
    const intWidth = parseInt(divStyles.width.replace("px", ""));
    const intHeight = parseInt(divStyles.height.replace("px", ""));

    const img = new Image();
    img.src = imageUrl;

    const canvas = document.createElement('canvas');
    canvas.width = intWidth;
    console.log(canvas.width, "vs", intWidth)

    canvas.height = intHeight;
    console.log(canvas.height, "vs", intHeight)


    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, intWidth, intHeight);

    const imageTensor = tf.browser.fromPixels(canvas);

    return imageTensor;

}

document.getElementById("inference-button").addEventListener("click", run);