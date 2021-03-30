import ai.djl.modality.nlp.embedding.TrainableWordEmbedding;
import ai.djl.ndarray.*;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.core.Linear;
import ai.djl.nn.recurrent.GRU;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public class Seq2SeqEncoder extends Decoder {

    private TrainableWordEmbedding embedding;
    private GRU rnn;

    public Seq2SeqEncoder(int vocabSize, int embedSize, int numHiddens, int numLayers, int dropout) {
        super();
        this.embedding = new TrainableWordEmbedding(new EmbedVocab(), embedSize);
        this.rnn =
                GRU.builder()
                        .setNumLayers(numLayers)
                        .setStateSize(numHiddens)
                        .optReturnState(true)
                        .optBatchFirst(false)
                        .optDropRate(dropout)
                        .build();
    }


    public NDList beginState(NDList encOutputs) {
        return new NDList(encOutputs.get(1));
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
