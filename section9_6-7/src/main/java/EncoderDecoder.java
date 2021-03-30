import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public class EncoderDecoder extends AbstractBlock {
    /* The base class for the encoder-decoder architecture. */
    private static final byte VERSION = 1;
    private Encoder encoder;
    private Decoder decoder;

    public EncoderDecoder(Encoder encoder, Decoder decoder) {
        super(VERSION);
        this.encoder = encoder;
        this.decoder = decoder;
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray encX = inputs.get(0);
        NDArray decX = inputs.get(1);
        NDList encOutputs = this.encoder.forward(parameterStore, new NDList(encX), training, params);
        return this.decoder.forward(parameterStore, new NDList(decX).addAll(encOutputs), training, params);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        throw new UnsupportedOperationException("Not implemented");
    }
}
