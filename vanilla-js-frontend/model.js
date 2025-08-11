import { loadImages } from './data.js';

function getCNNModel() {

    const model = tf.sequential();

    /*
    image sizes: not needed, all are resized to 512x512
    */
    const IMAGE_WIDTH = 512;
    const IMAGE_HEIGHT = 512;
    const IMAGE_DEPTH = 3; // RGB
    // const NUM_OUTPUT_CLASSES = 2;

    const optimizer = tf.train.adam();

    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

    model.add(tf.layers.flatten());

    model.add(tf.layers.dense({
        units: 1,
        kernelInitializer: 'varianceScaling',
        activation: 'sigmoid'
    }));

    model.compile({
        optimizer: optimizer,
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
    });

    return model;

} // getModel function

async function train(model, data) {
    
    const BATCH_SIZE = 3;
    const NUM_EPOCHS = 50;
    const TRAIN_DATA_SIZE = 12;
    const TEST_DATA_SIZE = 0;

    console.log("Fitting...");
    return model.fit(data[0] /* xs (aka features) */, 
        data[1] /* ys (aka labels) */, {
        batchSize: BATCH_SIZE,
        epochs: NUM_EPOCHS//, 
        // validationSplit: 0.2 // optional
    });
}

export {getCNNModel, train};