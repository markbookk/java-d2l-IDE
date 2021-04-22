import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public abstract class Decoder extends AbstractBlock {
    /* The base decoder interface for the encoder-decoder architecture. */
    private static final byte VERSION = 1;
    public NDArray attentionWeights;

    public Decoder() {
        super(VERSION);
    }

    @Override
    abstract protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params);

    abstract public NDList beginState(NDList encOutputs);

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        throw new UnsupportedOperationException("Not implemented");
    }
}
