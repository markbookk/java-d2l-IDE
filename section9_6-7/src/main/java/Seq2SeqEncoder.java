import ai.djl.modality.nlp.embedding.TrainableWordEmbedding;
import ai.djl.ndarray.*;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.nn.recurrent.GRU;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;
import ai.djl.util.PairList;

public class Seq2SeqEncoder extends Encoder {

    private TrainableWordEmbedding embedding;
    private GRU rnn;

    public Seq2SeqEncoder(
            int vocabSize, int embedSize, int numHiddens, int numLayers, float dropout) {
        super();
        this.embedding =
                TrainableWordEmbedding.builder()
                        .optNumEmbeddings(vocabSize)
                        .setEmbeddingSize(embedSize)
                        .setVocabulary(null)
                        .build();
//        this.embedding.setInitializer(Initializer.ZEROS, Parameter.Type.WEIGHT);
        this.addChildBlock("embedding", this.embedding);
        this.rnn =
                GRU.builder()
                        .setNumLayers(numLayers)
                        .setStateSize(numHiddens)
                        .optReturnState(true)
                        .optBatchFirst(false)
                        .optDropRate(dropout)
                        .build();
//        this.rnn.setInitializer(Initializer.ZEROS, Parameter.Type.WEIGHT);
        this.addChildBlock("rnn", this.rnn);
    }

    /** {@inheritDoc} */
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        return;
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray X = inputs.get(0);
        X = this.embedding.forward(parameterStore, new NDList(X), training, params).get(0);
        X = X.swapAxes(0, 1);

        return this.rnn.forward(parameterStore, new NDList(X), training);
    }
}
