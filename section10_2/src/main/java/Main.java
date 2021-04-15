import ai.djl.Model;
import ai.djl.ndarray.*;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.*;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.components.*;
import tech.tablesaw.plotly.traces.ScatterTrace;

import java.io.IOException;
import java.util.function.Function;

public class Main {

    public static NDManager manager;
    private static NDArray xTest;
    private static NDArray yTruth;
    private static NDArray xTrain;
    private static NDArray yTrain;

    public static void main(String[] args) throws IOException, TranslateException {
        manager = NDManager.newBaseManager(Functions.tryGpu(0));
        int nTrain = 50; // No. of training examples
        xTrain = manager.randomUniform(0, 1, new Shape(nTrain)).mul(5).sort(); // Training inputs

        Function<NDArray, NDArray> f = x -> x.sin().mul(2).add(x.pow(0.8));
        yTrain =
                f.apply(xTrain)
                        .add(
                                manager.randomNormal(
                                        0f,
                                        0.5f,
                                        new Shape(nTrain),
                                        DataType.FLOAT32)); // Training outputs
        xTest = manager.arange(0f, 5f, 0.1f); // Testing examples
        yTruth = f.apply(xTest); // Ground-truth outputs for the testing examples
        int nTest = (int) xTest.getShape().get(0); // No. of testing examples
        System.out.println(nTest);

        Figure fig =
                plot(
                        Functions.floatToDoubleArray(xTest.toFloatArray()),
                        Functions.floatToDoubleArray(yTruth.toFloatArray()),
                        Functions.floatToDoubleArray(yTrain.mean().tile(nTest).toFloatArray()),
                        Functions.floatToDoubleArray(xTrain.toFloatArray()),
                        Functions.floatToDoubleArray(yTrain.toFloatArray()),
                        "Truth",
                        "Pred",
                        "x",
                        "y",
                        700,
                        500);
        Plot.show(fig);

        // Shape of `xRepeat`: (`nTest`, `nTrain`), where each row contains the
        // same testing inputs (i.e., same queries)
        NDArray xRepeat = xTest.repeat(nTrain).reshape(new Shape(-1, nTrain));
        // Note that `xTrain` contains the keys. Shape of `attention_weights`:
        // (`nTest`, `nTrain`), where each row contains attention weights to be
        // assigned among the values (`yTrain`) given each query
        NDArray attentionWeights = xRepeat.sub(xTrain).pow(2).div(2).mul(-1).softmax(-1);
        // Each element of `yHat` is weighted average of values, where weights are
        // attention weights
        NDArray yHat = attentionWeights.dot(yTrain);
        fig =
                plot(
                        Functions.floatToDoubleArray(xTest.toFloatArray()),
                        Functions.floatToDoubleArray(yTruth.toFloatArray()),
                        Functions.floatToDoubleArray(yHat.toFloatArray()),
                        Functions.floatToDoubleArray(xTrain.toFloatArray()),
                        Functions.floatToDoubleArray(yTrain.toFloatArray()),
                        "Truth",
                        "Pred",
                        "x",
                        "y",
                        700,
                        500);
        Plot.show(fig);

        fig =
                PlotUtils.showHeatmaps(
                        attentionWeights.expandDims(0).expandDims(0),
                        "",
                        "",
                        new String[] {""},
                        500,
                        700);
        Plot.show(fig);

        NDArray X = manager.ones(new Shape(2, 1, 4));
        NDArray Y = manager.ones(new Shape(2, 4, 6));

        System.out.println(X.batchDot(Y).getShape());

        NDArray weights = manager.ones(new Shape(2, 10)).mul(0.1);
        NDArray values = manager.arange(20f).reshape(new Shape(2, 10));
        System.out.println(weights.expandDims(1).batchDot(values.expandDims(-1)));

        // Shape of `xTile`: (`nTrain`, `nTrain`), where each column contains the
        // same training inputs
        NDArray xTile = xTrain.tile(new long[] {nTrain, 1});
        // Shape of `Y_tile`: (`nTrain`, `nTrain`), where each column contains the
        // same training outputs
        NDArray yTile = yTrain.tile(new long[] {nTrain, 1});
        // Shape of `keys`: ('nTrain', 'nTrain' - 1)
        NDArray keys =
                xTile.get((manager.eye(nTrain).mul(-1).add(1))).reshape(new Shape(nTrain, -1));
        // Shape of `values`: ('nTrain', 'nTrain' - 1)
        values = yTile.get((manager.eye(nTrain).mul(-1).add(1))).reshape(new Shape(nTrain, -1));

        NWKernelRegression net = new NWKernelRegression();
        Loss loss = Loss.l2Loss();
        Tracker lrt =
                Tracker.fixed(0.5f * nTrain); // Since we are using sgd, to be able to put the right
        // scale, we need to multiply by batchSize
        Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();
        DefaultTrainingConfig config =
                new DefaultTrainingConfig(loss)
                        .optOptimizer(sgd) // Optimizer (loss function)
                        .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging
        Model model = Model.newInstance("");
        model.setBlock(net);

        Trainer trainer = model.newTrainer(config);

        // Create trainer and animator
        for (int epoch = 0; epoch < 5; epoch++) {
            try (GradientCollector gc = trainer.newGradientCollector()) {
                NDArray ressult =
                        net.forward(
                                        new ParameterStore(manager, false),
                                        new NDList(xTrain, keys, values),
                                        true)
                                .get(0);
                NDArray l = trainer.getLoss().evaluate(new NDList(yTrain), new NDList(ressult));

                gc.backward(l);
                System.out.println("Epoch: " + (epoch + 1) + " , Loss:" + l.mean().toString());
            }
            trainer.step();
        }

        // Shape of `keys`: (`nTest`, `nTrain`), where each column contains the same
        // training inputs (i.e., same keys)
        keys = xTrain.tile(new long[] {nTest, 1});

        // Shape of `value`: (`nTest`, `nTrain`)
        values = yTrain.tile(new long[] {nTest, 1});
        yHat =
                net.forward(
                                new ParameterStore(manager, false),
                                new NDList(xTest, keys, values),
                                true)
                        .get(0);
        fig =
                plot(
                        Functions.floatToDoubleArray(xTest.toFloatArray()),
                        Functions.floatToDoubleArray(yTruth.toFloatArray()),
                        Functions.floatToDoubleArray(yHat.toFloatArray()),
                        Functions.floatToDoubleArray(xTrain.toFloatArray()),
                        Functions.floatToDoubleArray(yTrain.toFloatArray()),
                        "Truth",
                        "Pred",
                        "x",
                        "y",
                        700,
                        500);
        Plot.show(fig);

        fig =
                PlotUtils.showHeatmaps(
                        net.attentionWeights.expandDims(0).expandDims(0),
                        "",
                        "",
                        new String[] {""},
                        500,
                        700);
        Plot.show(fig);
    }

    public static Figure plot(
            double[] xTest,
            double[] yTruth,
            double[] yHat,
            double[] xTrain,
            double[] yTrain,
            String trace1Name,
            String trace2Name,
            String xLabel,
            String yLabel,
            int width,
            int height) {
        ScatterTrace trace =
                ScatterTrace.builder(xTest, yTruth)
                        .mode(ScatterTrace.Mode.LINE)
                        .name(trace1Name)
                        .build();

        ScatterTrace trace2 =
                ScatterTrace.builder(xTest, yHat)
                        .mode(ScatterTrace.Mode.LINE)
                        .name(trace2Name)
                        .build();

        ScatterTrace trace3 =
                ScatterTrace.builder(xTrain, yTrain)
                        .mode(ScatterTrace.Mode.MARKERS)
                        .marker(Marker.builder().symbol(Symbol.CIRCLE).size(15).opacity(.5).build())
                        .build();

        Layout layout =
                Layout.builder()
                        .height(height)
                        .width(width)
                        .showLegend(true)
                        .xAxis(Axis.builder().title(xLabel).domain(0, 5).build())
                        .yAxis(Axis.builder().title(yLabel).domain(-1, 5).build())
                        .build();

        return new Figure(layout, trace, trace2, trace3);
    }
}
