import { loadImages } from 'data.js';

function getCNNModel() {

    model = tf.sequential();

    /*
    image sizes: 512x512, 512x512, 341x512, 341x512, 341x341
    */
    const IMAGE_WIDTH = 341;
    const IMAGE_HEIGHT = 341;
    const IMAGE_DEPTH = 3; // RGB
    // const NUM_OUTPUT_CLASSES = 2;

    const optimizer = tf.train.adam();
    
    // resize all images

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
        activation: 'softmax'
    }));

    model.compile({
        optimizer: optimizer,
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
    });

    return model;

} // getModel function

async function train(model, data) {
    
    const BATCH_SIZE = 512;
    const NUM_EPOCHS = 10;
    const TRAIN_DATA_SIZE = 12;
    const TEST_DATA_SIZE = 0;

    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        epochs: NUM_EPOCHS//, 
        // validationSplit: 0.2 // optional
    });
}