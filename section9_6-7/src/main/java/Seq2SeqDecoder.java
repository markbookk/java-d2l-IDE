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

public class Seq2SeqDecoder extends Decoder {

    private TrainableWordEmbedding embedding;
    private GRU rnn;
    private Linear dense;

    public Seq2SeqDecoder(
            int vocabSize, int embedSize, int numHiddens, int numLayers, float dropout) {
        super();
        this.embedding =
                TrainableWordEmbedding.builder()
                        .optNumEmbeddings(vocabSize)
                        .setEmbeddingSize(embedSize)
                        .setVocabulary(null)
                        .build();
//        this.embedding.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
        this.addChildBlock("embedding", this.embedding);

        this.rnn =
                GRU.builder()
                        .setNumLayers(numLayers)
                        .setStateSize(numHiddens)
                        .optReturnState(true)
                        .optBatchFirst(false)
                        .optDropRate(dropout)
                        .build();
//        this.rnn.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
        this.addChildBlock("rnn", this.rnn);

        this.dense = Linear.builder().setUnits(vocabSize).build();
//        this.dense.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
        this.addChildBlock("dense", this.dense);
    }

    /** {@inheritDoc} */
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        return;
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
