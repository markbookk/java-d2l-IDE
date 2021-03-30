import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public abstract class Encoder extends AbstractBlock {
    /* The base encoder interface for the encoder-decoder architecture. */
    private static final byte VERSION = 1;

    public Encoder() {
        super(VERSION);
    }

    @Override
    abstract protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params);

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        throw new UnsupportedOperationException("Not implemented");
    }
}
