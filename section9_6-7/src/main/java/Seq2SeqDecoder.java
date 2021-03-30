import ai.djl.modality.nlp.SimpleVocabulary;
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding;
import ai.djl.ndarray.*;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Embedding;
import ai.djl.nn.core.Linear;
import ai.djl.nn.recurrent.GRU;
import ai.djl.nn.recurrent.RNN;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public class Seq2SeqDecoder extends Decoder {

    private TrainableWordEmbedding embedding;
    private GRU rnn;
    private Linear dense;

    public Seq2SeqDecoder(int vocabSize, int embedSize, int numHiddens, int numLayers, int dropout) {
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
        this.dense = Linear.builder().setUnits(vocabSize).build();
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
        NDArray state = inputs.get(1);
        X =
                this.embedding
                        .forward(parameterStore, new NDList(X), training, params)
                        .get(0)
                        .swapAxes(0, 1);
        NDArray context = inputs.get(1).get(new NDIndex(-1));
        context =
                context.broadcast(
                        new Shape(
                                X.getShape().get(0),
                                context.getShape().get(0),
                                context.getShape().get(1)));
        NDArray xAndContext = NDArrays.concat(new NDList(X, context), 2);
        NDList rnnOutput =
                this.rnn.forward(parameterStore, new NDList(xAndContext, state), training);
        NDArray output = rnnOutput.get(0);
        state = rnnOutput.get(1);
        output =
                this.dense
                        .forward(parameterStore, new NDList(output), training)
                        .get(0)
                        .swapAxes(0, 1);
        return new NDList(output, state);
    }
}
