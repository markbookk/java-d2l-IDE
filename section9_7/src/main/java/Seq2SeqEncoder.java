import ai.djl.nn.core.Embedding;
import ai.djl.nn.recurrent.RecurrentBlock;

import java.beans.Encoder;

public class Seq2SeqEncoder extends Encoder {
    Embedding<Integer> embedding;

    public Seq2SeqEncoder(int vocabSize, int embedSize, int numHiddens, int numLayers, int dropout) {
        this.embedding = new Embedding.BaseBuilder<Integer, >();
    }
}
