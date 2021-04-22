import ai.djl.modality.nlp.embedding.TrainableWordEmbedding;
import ai.djl.ndarray.*;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.nn.recurrent.GRU;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;
import ai.djl.util.Pair;
import ai.djl.util.PairList;

public class Seq2SeqAttentionDecoder extends AttentionDecoder {
    private AdditiveAttention attention;
    private TrainableWordEmbedding embedding;
    private GRU rnn;
    private Linear dense;
    private NDList attentionWeights;

    public Seq2SeqAttentionDecoder(
            int vocabSize, int embedSize, int numHiddens, int numLayers, float dropout) {
        super();
        this.attention = new AdditiveAttention(numHiddens, dropout);
        this.addChildBlock("attention", this.attention);

        this.embedding =
                TrainableWordEmbedding.builder()
                        .optNumEmbeddings(vocabSize)
                        .setEmbeddingSize(embedSize)
                        .setVocabulary(null)
                        .build();
        this.addChildBlock("embedding", this.embedding);

        this.rnn =
                GRU.builder()
                        .setNumLayers(numLayers)
                        .setStateSize(numHiddens)
                        .optReturnState(true)
                        .optBatchFirst(false)
                        .optDropRate(dropout)
                        .build();

        this.dense = Linear.builder().setUnits(vocabSize).build();
        this.dense.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
        this.addChildBlock("dense", this.dense);
    }

    public NDList beginState(NDList inputs) {
        // Output, hiddenState, encValidLens
        NDList encOutputs = new NDList(inputs.subList(0, inputs.getShapes().length - 1));
        NDArray encValidLens = inputs.get(inputs.getShapes().length - 1);
        NDArray outputs = encOutputs.get(0).swapAxes(0, 1);
        NDList hiddenState = encOutputs.subNDList(1);
        NDList temp = new NDList(outputs).addAll(hiddenState);
        temp.add(encValidLens);
        return temp;
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray X = inputs.get(0);
        NDList state = inputs.subNDList(1);

        NDArray encOutputs = state.get(0); // First element
        NDList hiddenState =
                new NDList(
                        state.subList(
                                1,
                                state.getShapes().length
                                        - 1)); // From second element to second to last element
        NDArray encValidLens = state.get(state.getShapes().length - 1); // Last element

        X =
                this.embedding
                        .forward(parameterStore, new NDList(X), training, params)
                        .get(0)
                        .swapAxes(0, 1);
        NDList outputs = new NDList();
        this.attentionWeights = new NDList();
        for (int i = 0; i < X.getShape().get(0); i++) {
            NDArray query = hiddenState.get(0).get(-1).expandDims(1);
            NDArray context =
                    this.attention
                            .forward(
                                    parameterStore,
                                    new NDList(query, encOutputs, encOutputs, encValidLens),
                                    training,
                                    params)
                            .get(0);
            NDArray x = NDArrays.concat(new NDList(context, X.get(i).expandDims(1)), -1);
            NDList output =
                    this.rnn.forward(
                            parameterStore,
                            (new NDList(x.swapAxes(0, 1))).addAll(hiddenState),
                            training,
                            params);
            outputs.add(output.get(0));
            hiddenState = output.subNDList(1);
            this.attentionWeights.add(this.attention.attentionWeights);
        }
        outputs =
                this.dense.forward(
                        parameterStore, new NDList(NDArrays.concat(outputs, 0)), training, params);
        NDList temp = new NDList(outputs.get(0).swapAxes(0, 1));
        temp.add(encOutputs);
        temp = temp.addAll(hiddenState);
        temp.add(encValidLens);
        return temp;
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[0];
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {}
}
