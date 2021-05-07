import ai.djl.ndarray.*;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.UniformInitializer;
import ai.djl.util.PairList;

/* Additive attention. */
public class AdditiveAttention extends AbstractBlock {
    private static final byte VERSION = 1;
    private Linear W_k;
    private Linear W_q;
    private Linear W_v;
    private Dropout dropout;
    public NDArray attentionWeights;

    public AdditiveAttention(int numHiddens, float dropout) {
        super(VERSION);
        this.W_k = Linear.builder().setUnits(numHiddens).optBias(false).build();
        this.W_k.setInitializer(new UniformInitializer(0.07f), Parameter.Type.WEIGHT);
        this.W_k.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
        this.addChildBlock("W_k", this.W_k);

        this.W_q = Linear.builder().setUnits(numHiddens).optBias(false).build();
        this.W_q.setInitializer(new UniformInitializer(0.07f), Parameter.Type.WEIGHT);
        this.W_q.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
        this.addChildBlock("W_q", this.W_q);

        this.W_v = Linear.builder().setUnits(1).optBias(false).build();
        this.W_v.setInitializer(new UniformInitializer(0.07f), Parameter.Type.WEIGHT);
        this.W_v.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
        this.addChildBlock("W_v", this.W_v);

        this.dropout = Dropout.builder().optRate(dropout).build();
        this.dropout.setInitializer(new UniformInitializer(0.07f), Parameter.Type.WEIGHT);
        this.dropout.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
        this.addChildBlock("dropout", this.dropout);
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
        NDArray validLens = inputs.get(3);

        queries = this.W_q.forward(parameterStore, new NDList(queries), training, params).get(0);
        keys = this.W_k.forward(parameterStore, new NDList(keys), training, params).get(0);
        // After dimension expansion, shape of `queries`: (`batchSize`, no. of
        // queries, 1, `numHiddens`) and shape of `keys`: (`batchSize`, 1,
        // no. of key-value pairs, `numHiddens`). Sum them up with
        // broadcasting
        NDArray features = queries.expandDims(2).add(keys.expandDims(1));
        features = features.tanh();
        // There is only one output of `this.W_v`, so we remove the last
        // one-dimensional entry from the shape. Shape of `scores`:
        // (`batchSize`, no. of queries, no. of key-value pairs)
        NDArray result =
                this.W_v.forward(parameterStore, new NDList(features), training, params).get(0);
        NDArray scores = result.squeeze(-1);
        this.attentionWeights = Chap10Utils.maskedSoftmax(scores, validLens);
        // Shape of `values`: (`batchSize`, no. of key-value pairs, value
        // dimension)
        return new NDList(
                this.dropout
                        .forward(
                                parameterStore, new NDList(this.attentionWeights), training, params)
                        .get(0)
                        .batchDot(values));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {}
}
