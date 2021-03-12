import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.recurrent.RNN;
import ai.djl.training.ParameterStore;

public class Main {
    public static NDManager manager;

    public static void main(String[] args) throws Exception {
        manager = NDManager.newBaseManager(Functions.tryGpu(0));

        int batchSize = 32;
        int numSteps = 35;

//        Pair<ArrayList<NDList>, Vocab> timeMachine =
//                SeqDataLoader.loadDataTimeMachine(batchSize, numSteps, true, 10000, manager);
//        ArrayList<NDList> trainIter = timeMachine.getKey();
//        Vocab vocab = timeMachine.getValue();
        TimeMachineDataset dataset = new TimeMachineDataset.Builder()
                .setManager(manager).setMaxTokens(10000).setSampling(batchSize, false)
                .setSteps(numSteps).build();
        dataset.prepare();
        Vocab vocab = dataset.getVocab();

        int numHiddens = 256;
        RNN rnnLayer = RNN.builder().setNumLayers(1)
                .setStateSize(numHiddens).optReturnState(true).optBatchFirst(false).build();
//        rnnLayer.setInitializer(Initializer.ZEROS, Parameter.Type.WEIGHT);
        rnnLayer.initialize(manager, DataType.FLOAT32, new Shape(numSteps, batchSize, vocab.length()));


        NDList state = beginState(batchSize, 1, numHiddens);
        System.out.println(state.size());
        System.out.println(state.get(0).getShape());

        NDArray X = manager.randomUniform(0, 1,new Shape(numSteps, batchSize, vocab.length()));
        X = manager.zeros(new Shape(numSteps, batchSize, vocab.length()));

        NDList forwardOutput = rnnLayer.forward(new ParameterStore(manager, false), new NDList(X, state.get(0)), false);
        NDArray Y = forwardOutput.get(0);
        NDArray stateNew = forwardOutput.get(1);

        System.out.println(Y.getShape());
        System.out.println(stateNew.getShape());

        Device device = Functions.tryGpu(0);
        RNNModel net = new RNNModel(rnnLayer, vocab.length());
        net.initialize(manager, DataType.FLOAT32, X.getShape());
        String prediction = TimeMachine.predictCh8("time traveller", 10, net, vocab, device, manager);
        System.out.println(prediction);

        int numEpochs = 500;
        int lr = 1;
        TimeMachine.trainCh8((Object) net, dataset, vocab, lr, numEpochs, device, false, manager);

        System.out.println("debug");
    }

    public static NDList beginState(int batchSize, int numLayers, int numHiddens) {
        return new NDList (manager.zeros(new Shape(numLayers, batchSize, numHiddens)));
    }
}