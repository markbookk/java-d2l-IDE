import ai.djl.ndarray.*;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.translate.TranslateException;
import tech.tablesaw.index.Index;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.components.Figure;

import java.io.IOException;

public class Main {

    public static NDManager manager;

    public static void main(String[] args) throws IOException, TranslateException {
        manager = NDManager.newBaseManager(Functions.tryGpu(0));

        int numHiddens = 100;
        int numHeads = 5;
        MultiHeadAttention attention = new MultiHeadAttention(numHiddens, numHeads, 0.5f, false);

        int batchSize = 2;
        int numQueries = 4;
        NDArray validLens = manager.create(new float[] {3, 2});
        NDArray X = manager.ones(new Shape(batchSize, numQueries, numHiddens));
        System.out.println(
                attention
                        .forward(
                                new ParameterStore(manager, false),
                                new NDList(X, X, X, validLens),
                                false)
                        .get(0)
                        .getShape());

        int encodingDim = 32;
        int numSteps = 60;
        PositionalEncoding posEncoding = new PositionalEncoding(encodingDim, 0, 1000, manager);
        X =
                posEncoding
                        .forward(
                                new ParameterStore(manager, false),
                                new NDList(manager.zeros(new Shape(1, numSteps, encodingDim))),
                                false)
                        .get(0);
        NDArray P = posEncoding.P.get(new NDIndex(":, :{}, :", X.getShape().get(1)));

        double[][] plotX = new double[4][];
        double[][] plotY = new double[4][];
        for (int i = 0; i < 4; i++) {
            if (i == 0) {
                plotX[i] = manager.arange(numSteps).toType(DataType.FLOAT64, false).toDoubleArray();
            } else {
                plotX[i] = plotX[i - 1];
            }
            plotY[i] =
                    Functions.floatToDoubleArray(
                            P.get(new NDIndex("0, :, {},", i + 6)).toFloatArray());
        }

        Figure fig =
                PlotUtils.plot(
                        plotX,
                        plotY,
                        new String[] {"Col6", "Col7", "Col8", "Col9"},
                        "Row (position)",
                        "");
        Plot.show(fig);

        for (int i = 0; i < 8; i++) {
            System.out.println(i + " in binary is " + Integer.toBinaryString(i));
        }
        P = P.get(new NDIndex("0, :, :")).expandDims(0).expandDims(0);
        fig = PlotUtils.showHeatmaps(
                P, "Column (encoding dimension)", "Row (position)", new String[] {""}, 500, 700);
        Plot.show(fig);
    }

    public static NDArray transposeOutput(NDArray X, int numHeads) {
        X = X.reshape(-1, numHeads, X.getShape().get(1), X.getShape().get(2));
        X = X.transpose(0, 2, 1, 3);
        return X.reshape(X.getShape().get(0), X.getShape().get(1), -1);
    }
}
