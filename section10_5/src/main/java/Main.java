import ai.djl.ndarray.*;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.translate.TranslateException;
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
        int numKvpairs = 6;
        NDArray validLens = manager.create(new float[] {3, 2});
        NDArray X = manager.ones(new Shape(batchSize, numQueries, numHiddens));
        NDArray Y = manager.ones(new Shape(batchSize, numKvpairs, numHiddens));
        System.out.println(
                attention
                        .forward(
                                new ParameterStore(manager, false),
                                new NDList(X, Y, Y, validLens),
                                false)
                        .get(0)
                        .getShape());
    }
}
