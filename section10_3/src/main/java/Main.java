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

        System.out.println(
                maskedSoftmax(
                        manager.randomUniform(0, 1, new Shape(2, 2, 4)),
                        manager.create(new float[] {2, 3})));

        System.out.println(
                maskedSoftmax(
                        manager.randomUniform(0, 1, new Shape(2, 2, 4)),
                        manager.create(new float[][] {{1, 3}, {2, 4}})));

        NDArray queries = manager.randomNormal(0, 1, new Shape(2, 1, 20), DataType.FLOAT32);
        NDArray keys = manager.ones(new Shape(2, 10, 2));
        // The two value matrices in the `values` minibatch are identical
        NDArray values = manager.arange(40f).reshape(1, 10, 4).repeat(0, 2);
        NDArray validLens = manager.create(new float[] {2, 6});

        AdditiveAttention attention = new AdditiveAttention(8, 0.1f);
        attention
                .forward(
                        new ParameterStore(manager, false),
                        new NDList(queries, keys, values, validLens),
                        false)
                .get(0);

        Figure fig = PlotUtils.showHeatmaps(
                attention.attentionWeights.reshape(1, 1, 2, 10),
                "Keys",
                "Queries",
                new String[] {""},
                500,
                700);
        Plot.show(fig);

        queries = manager.randomNormal(0, 1, new Shape(2, 1, 2), DataType.FLOAT32);
        DotProductAttention productAttention = new DotProductAttention(0.5f);
        productAttention
                .forward(
                        new ParameterStore(manager, false),
                        new NDList(queries, keys, values, validLens),
                        false)
                .get(0);

        fig = PlotUtils.showHeatmaps(
                productAttention.attentionWeights.reshape(1, 1, 2, 10),
                "Keys",
                "Queries",
                new String[] {""},
                500,
                700);
        Plot.show(fig);
    }

    public static NDArray maskedSoftmax(NDArray X, NDArray validLens) {
        /* Perform softmax operation by masking elements on the last axis. */
        // `X`: 3D tensor, `validLens`: 1D or 2D tensor
        if (validLens == null) {
            return X.softmax(-1);
        } else {
            Shape shape = X.getShape();
            if (validLens.getShape().dimension() == 1) {
                validLens = validLens.repeat(shape.get(1));
            } else {
                validLens = validLens.reshape(-1);
            }
            // On the last axis, replace masked elements with a very large negative
            // value, whose exponentiation outputs 0
            X =
                    X.reshape(new Shape(-1, shape.get(shape.dimension() - 1)))
                            .sequenceMask(validLens, (float) -1E6);
            return X.softmax(-1).reshape(shape);
        }
    }
}
