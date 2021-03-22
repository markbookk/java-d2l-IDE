import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;

import java.io.IOException;

public class Main {

    public static NDManager manager;

    public static void main(String[] args) throws IOException, TranslateException {
        manager = NDManager.newBaseManager(Functions.tryGpu(0));
        // Load data
        int batchSize = 32;
        int numSteps = 35;
        Device device = Functions.tryGpu(0);
        TimeMachineDataset dataset =
                new TimeMachineDataset.Builder()
                        .setManager(manager)
                        .setMaxTokens(10000)
                        .setSampling(batchSize, false)
                        .setSteps(numSteps)
                        .build();
        dataset.prepare();
        Vocab vocab = dataset.getVocab();

        // Define the bidirectional LSTM model by setting `bidirectional=True`
        int vocabSize = vocab.length();
        int numHiddens = 256;
        int numLayers = 2;
        LSTM lstmLayer =
                LSTM.builder()
                        .setNumLayers(numLayers)
                        .setStateSize(numHiddens)
                        .optReturnState(true)
                        .optBatchFirst(false)
                        .optBidirectional(true)
                        .build();

        // Train the model
        RNNModel model = new RNNModel(lstmLayer, vocabSize);
        int numEpochs = 500;
        int lr = 1;
        TimeMachine.trainCh8(model, dataset, vocab, lr, numEpochs, device, false, manager);
    }
}
