import ai.djl.nn.AbstractBlock;

/* The base attention-based decoder interface. */
public abstract class AttentionDecoder extends Decoder {
    public AttentionDecoder() {
        super();
    }

    public void attentionWeights() {
        throw new UnsupportedOperationException("Not implemented");
    }
}
