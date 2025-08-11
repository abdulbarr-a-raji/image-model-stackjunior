import { getCNNModel, train } from './model.js';
import { loadImages } from './data.js';

async function runMain () {

    console.log("Button clicked...");

    const trainingData = await loadImages();
    /* below code should be moved to a new all-encompassing 
    'run' function
    */
    const convnet = getCNNModel();

    train(convnet, trainingData);
    console.log("Training complete...!");

    const savedModel = await convnet.save("downloads://pretrained-model-v1");

}

document.getElementById("load-images").addEventListener("click", runMain);