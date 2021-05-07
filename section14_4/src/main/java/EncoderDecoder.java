import ai.djl.ndarray.*;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;
import ai.djl.util.PairList;

public class EncoderDecoder extends AbstractBlock {
    /* The base class for the encoder-decoder architecture. */
    private static final byte VERSION = 1;
    public Encoder encoder;
    public Decoder decoder;

    public EncoderDecoder(Encoder encoder, Decoder decoder) {
        super(VERSION);

        this.encoder = encoder;
        this.encoder.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
        this.addChildBlock("encoder", this.encoder);
        this.decoder = decoder;
        this.decoder.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
        this.addChildBlock("decoder", this.decoder);
    }

    /** {@inheritDoc} */
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {}

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray encX = inputs.get(0);
        NDArray decX = inputs.get(1);
        NDArray lenX = inputs.get(2);
        NDList encOutputs =
                this.encoder.forward(parameterStore, new NDList(encX), training, params);
        encOutputs.add(lenX);
        NDList decState = this.decoder.beginState(encOutputs);
        return this.decoder.forward(
                parameterStore, new NDList(decX).addAll(decState), training, params);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        throw new UnsupportedOperationException("Not implemented");
    }
}
