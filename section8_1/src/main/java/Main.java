import ai.djl.Device;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.util.Pair;
import tech.tablesaw.plotly.components.Axis;
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.traces.ScatterTrace;

import java.io.IOException;

public class Main {

    public static NDManager manager;

    public static void main(String[] args) throws IOException, TranslateException {
        manager = NDManager.newBaseManager(Functions.tryGpu(0));

        int T = 1000; //Generate a total of 1000 points
        NDArray time = manager.arange(1f, T+1);
        NDArray x = time.mul(0.01).sin().add(
                manager.randomNormal(0f, 0.2f, new Shape(T), DataType.FLOAT32));

        double[] xAxis = Functions.floatToDoubleArray(time.toFloatArray());
        double[] yAxis = Functions.floatToDoubleArray(x.toFloatArray());

        plot(xAxis, yAxis, "time", "x");

        int tau = 4;
        NDArray features = manager.zeros(new Shape(T - tau, tau));

        for (int i = 0; i < tau; i++) {
            features.set(new NDIndex(":, {}", i), x.get(new NDIndex("{}:{}", i, T - tau + i)));
        }
        NDArray labels = x.get(new NDIndex("" + tau + ":")).reshape(new Shape(-1,1));

        int batchSize = 16;
        int nTrain = 600;
        // Only the first `nTrain` examples are used for training
        ArrayDataset trainIter = new ArrayDataset.Builder()
                .setData(features.get(new NDIndex(":{}", nTrain)))
                .optLabels(labels.get(new NDIndex(":{}", nTrain)))
                .setSampling(batchSize, true)
                .build();

        SequentialBlock net = getNet();
        Model model = train(net, trainIter, batchSize, 5, 0.01f);

        Translator translator = new NoopTranslator(null);
        Predictor predictor = model.newPredictor(translator);

        NDArray onestepPreds = ((NDList) predictor.predict(new NDList(features))).get(0);
        System.out.println(onestepPreds);

        ScatterTrace trace = ScatterTrace.builder(Functions.floatToDoubleArray(time.toFloatArray()),
                Functions.floatToDoubleArray(x.toFloatArray()))
                .mode(ScatterTrace.Mode.LINE)
                .name("data")
                .build();

        ScatterTrace trace2 = ScatterTrace.builder(Functions.floatToDoubleArray(time.get(new NDIndex("{}:", tau)).toFloatArray()),
                Functions.floatToDoubleArray(onestepPreds.toFloatArray()))
                .mode(ScatterTrace.Mode.LINE)
                .name("1-step preds")
                .build();

        Layout layout = Layout.builder()
                .showLegend(true)
                .xAxis(Axis.builder().title("time").build())
                .yAxis(Axis.builder().title("x").build())
                .build();

        new Figure(layout, trace, trace2);

        NDArray multiStepPreds = manager.zeros(new Shape(T));
        multiStepPreds.set(new NDIndex(":{}", nTrain + tau), x.get(new NDIndex(":{}", nTrain + tau)));
        for (int i = nTrain + tau; i < T; i++) {
            NDArray tempX = multiStepPreds.get(new NDIndex("{}:{}", i - tau, i)).reshape(new Shape(1, -1));
            NDArray prediction = ((NDList) predictor.predict(new NDList(tempX))).get(0);
            multiStepPreds.set(new NDIndex(i), prediction);
        }

        trace = ScatterTrace.builder(Functions.floatToDoubleArray(time.toFloatArray()),
                Functions.floatToDoubleArray(x.toFloatArray()))
                .mode(ScatterTrace.Mode.LINE)
                .name("data")
                .build();

        trace2 = ScatterTrace.builder(Functions.floatToDoubleArray(time.get(new NDIndex("{}:", tau)).toFloatArray()),
                Functions.floatToDoubleArray(onestepPreds.toFloatArray()))
                .mode(ScatterTrace.Mode.LINE)
                .name("1-step preds")
                .build();

        ScatterTrace trace3 = ScatterTrace.builder(Functions.floatToDoubleArray(time.get(
                new NDIndex("{}:", nTrain + tau)).toFloatArray()),
                Functions.floatToDoubleArray(multiStepPreds.get(
                        new NDIndex("{}:", nTrain + tau)).toFloatArray()))
                .mode(ScatterTrace.Mode.LINE)
                .name("multistep preds")
                .build();

        layout = Layout.builder()
                .showLegend(true)
                .xAxis(Axis.builder().title("time").build())
                .yAxis(Axis.builder().title("x").build())
                .build();

        new Figure(layout, trace, trace2, trace3);


    }

    public static Model train(SequentialBlock net, ArrayDataset dataset, int batchSize, int numEpochs, float learningRate)
            throws IOException, TranslateException {
        // Square Loss
        Loss loss = Loss.l2Loss();
        Tracker lrt = Tracker.fixed(learningRate);
        Optimizer adam = Optimizer.adam().optLearningRateTracker(lrt).build();

        DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
                .optOptimizer(adam) // Optimizer (loss function)
                .optInitializer(new XavierInitializer())
                .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

        Model model = Model.newInstance("sequence");
        model.setBlock(net);
        Trainer trainer = model.newTrainer(config);

        for (int epoch = 1; epoch <= numEpochs; epoch++) {
            // Iterate over dataset
            for (Batch batch : trainer.iterateDataset(dataset)) {
                // Update loss and evaulator
                EasyTrain.trainBatch(trainer, batch);

                // Update parameters
                trainer.step();

                batch.close();
            }

            // reset training and validation evaluators at end of epoch
            trainer.notifyListeners(listener -> listener.onEpoch(trainer));
            System.out.printf("Epoch %d\n", epoch);
            System.out.printf("Loss %f\n", trainer.getTrainingResult().getTrainLoss());
        }
        return model;
    }


    // A simple MLP
    public static SequentialBlock getNet() {
        SequentialBlock net = new SequentialBlock();
        net.add(Linear.builder().setUnits(10).build());
        net.add(Activation::relu);
        net.add(Linear.builder().setUnits(1).build());
        return net;
    }

    public static Figure plot(double[] x, double[] y, String xLabel, String yLabel) {
        ScatterTrace trace = ScatterTrace.builder(x,y)
                .mode(ScatterTrace.Mode.LINE)
                .build();

        Layout layout = Layout.builder()
                .showLegend(true)
                .xAxis(Axis.builder().title(xLabel).build())
                .yAxis(Axis.builder().title(yLabel).build())
                .build();

        return new Figure(layout, trace);
    }
}
