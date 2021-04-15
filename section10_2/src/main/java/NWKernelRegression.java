import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.UniformInitializer;
import ai.djl.util.PairList;

public class NWKernelRegression extends AbstractBlock {
    private static final byte VERSION = 1;
    private static Parameter w;
    public NDArray attentionWeights;

    public NWKernelRegression() {
        super(VERSION);
        this.w = Parameter.builder().optShape(new Shape(1)).setName("w").optInitializer(new UniformInitializer()).build();
        this.addParameter(this.w);
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        // Shape of the output `queries` and `attentionWeights`:
        // (no. of queries, no. of key-value pairs)
        NDArray queries = inputs.get(0);
        NDArray keys = inputs.get(1);
        NDArray values = inputs.get(2);
        queries =
                queries.repeat(keys.getShape().get(1))
                        .reshape(new Shape(-1, keys.getShape().get(1)));

        this.attentionWeights =
                queries.sub(keys).mul(this.w.getArray()).pow(2).div(2).mul(-1).softmax(-1);
        // Shape of `values`: (no. of queries, no. of key-value pairs)
        return new NDList(
                this.attentionWeights.expandDims(1).batchDot(values.expandDims(-1)).reshape(-1));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        throw new UnsupportedOperationException("Not implemented");
    }
}
