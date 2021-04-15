import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.TranslateException;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.components.*;
import tech.tablesaw.plotly.traces.HeatmapTrace;
import tech.tablesaw.plotly.traces.Trace;

import java.io.IOException;

public class Main {

    public static NDManager manager;

    public static void main(String[] args) throws IOException, TranslateException {
        manager = NDManager.newBaseManager(Functions.tryGpu(0));
        NDArray attentionWeights = manager.eye(10).reshape(new Shape(1, 1, 10, 10));

        Figure fig = showHeatmaps(attentionWeights, "Keys", "Queries", null, 700, 1000);
        Plot.show(fig);
    }

    public static Figure showHeatmaps(
            NDArray matrices,
            String xLabel,
            String yLabel,
            String[] titles,
            int width,
            int height) {
        int numRows = (int) matrices.getShape().get(0);
        int numCols = (int) matrices.getShape().get(1);

        Trace[] traces = new Trace[numRows * numCols];
        int count = 0;
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                NDArray NDMatrix = matrices.get(i).get(j);
                double[][] matrix =
                        new double[(int) NDMatrix.getShape().get(0)]
                                [(int) NDMatrix.getShape().get(1)];
                Object[] x = new Object[matrix.length];
                Object[] y = new Object[matrix.length];
                for (int k = 0; k < NDMatrix.getShape().get(0); k++) {
                    matrix[k] = Functions.floatToDoubleArray(NDMatrix.get(k).toFloatArray());
                    x[k] = k;
                    y[k] = k;
                }
                HeatmapTrace.HeatmapBuilder builder = HeatmapTrace.builder(x, y, matrix);
                if (titles != null) {
                    builder = (HeatmapTrace.HeatmapBuilder) builder.name(titles[j]);
                }
                traces[count++] = builder.build();
            }
        }
        Grid grid =
                Grid.builder()
                        .columns(numCols)
                        .rows(numRows)
                        .pattern(Grid.Pattern.INDEPENDENT)
                        .build();
        Layout layout =
                Layout.builder()
                        .title("")
                        .xAxis(Axis.builder().title(xLabel).build())
                        .yAxis(Axis.builder().title(yLabel).build())
                        .width(width)
                        .height(height)
                        .grid(grid)
                        .build();
        return new Figure(layout, traces);
    }
}
