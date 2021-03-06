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
        int batchSize = 32;
        int numSteps = 35;

        TimeMachineDataset dataset =
                new TimeMachineDataset.Builder()
                        .setManager(manager)
                        .setMaxTokens(10000)
                        .setSampling(batchSize, false)
                        .setSteps(numSteps)
                        .build();
        dataset.prepare();
        Vocab vocab = dataset.getVocab();

        int vocabSize = vocab.length();
        int numHiddens = 256;
        Device device = Functions.tryGpu(0);
        int numEpochs = 500;
        int lr = 1;

        Functions.TriFunction<Integer, Integer, Device, NDList> getParamsFn =
                (a, b, c) -> getLSTMParams(a, b, c);
        Functions.TriFunction<Integer, Integer, Device, NDList> initLSTMStateFn =
                (a, b, c) -> initLSTMState(a, b, c);
        Functions.TriFunction<NDArray, NDList, NDList, Pair> lstmFn = (a, b, c) -> lstm(a, b, c);

        RNNModelScratch model =
                new RNNModelScratch(
                        vocabSize, numHiddens, device, getParamsFn, initLSTMStateFn, lstmFn);
//        TimeMachine.trainCh8(model, dataset, vocab, lr, numEpochs, device, false, manager);

        LSTM lstmLayer =
                LSTM.builder()
                        .setNumLayers(1)
                        .setStateSize(numHiddens)
                        .optReturnState(true)
                        .optBatchFirst(false)
                        .build();
        RNNModel modelConcise = new RNNModel(lstmLayer, vocab.length());
        TimeMachine.trainCh8(modelConcise, dataset, vocab, lr, numEpochs, device, false, manager);
    }

    public static NDList initLSTMState(int batchSize, int numHiddens, Device device) {
        return new NDList(
                manager.zeros(new Shape(batchSize, numHiddens), DataType.FLOAT32, device),
                manager.zeros(new Shape(batchSize, numHiddens), DataType.FLOAT32, device));
    }

    public static Pair<NDArray, NDList> lstm(NDArray inputs, NDList state, NDList params) {
        NDArray W_xi = params.get(0);
        NDArray W_hi = params.get(1);
        NDArray b_i = params.get(2);

        NDArray W_xf = params.get(3);
        NDArray W_hf = params.get(4);
        NDArray b_f = params.get(5);

        NDArray W_xo = params.get(6);
        NDArray W_ho = params.get(7);
        NDArray b_o = params.get(8);

        NDArray W_xc = params.get(9);
        NDArray W_hc = params.get(10);
        NDArray b_c = params.get(11);

        NDArray W_hq = params.get(12);
        NDArray b_q = params.get(13);

        NDArray H = state.get(0);
        NDArray C = state.get(1);
        NDList outputs = new NDList();
        NDArray X, Y, I, F, O, C_tilda;
        for (int i = 0; i < inputs.size(0); i++) {
            X = inputs.get(i);
            I = Activation.sigmoid(X.dot(W_xi).add(H.dot(W_hi).add(b_i)));
            F = Activation.sigmoid(X.dot(W_xf).add(H.dot(W_hf).add(b_f)));
            O = Activation.sigmoid(X.dot(W_xo).add(H.dot(W_ho).add(b_o)));
            C_tilda = Activation.tanh(X.dot(W_xc).add(H.dot(W_hc).add(b_c)));
            C = F.mul(C).add(I.mul(C_tilda));
            H = O.mul(Activation.tanh(C));
            Y = H.dot(W_hq).add(b_q);
            outputs.add(Y);
        }
        return new Pair(
                outputs.size() > 1 ? NDArrays.concat(outputs) : outputs.get(0), new NDList(H, C));
    }

    public static NDList getLSTMParams(int vocabSize, int numHiddens, Device device) {
        int numInputs = vocabSize;
        int numOutputs = vocabSize;

        // Input gate parameters
        NDList temp = three(numInputs, numHiddens, device);
        NDArray W_xi = temp.get(0);
        NDArray W_hi = temp.get(1);
        NDArray b_i = temp.get(2);

        // Forget gate parameters
        temp = three(numInputs, numHiddens, device);
        NDArray W_xf = temp.get(0);
        NDArray W_hf = temp.get(1);
        NDArray b_f = temp.get(2);

        // Output gate parameters
        temp = three(numInputs, numHiddens, device);
        NDArray W_xo = temp.get(0);
        NDArray W_ho = temp.get(1);
        NDArray b_o = temp.get(2);

        // Candidate memory cell parameters
        temp = three(numInputs, numHiddens, device);
        NDArray W_xc = temp.get(0);
        NDArray W_hc = temp.get(1);
        NDArray b_c = temp.get(2);

        // Output layer parameters
        NDArray W_hq = normal(new Shape(numHiddens, numOutputs), device);
        NDArray b_q = manager.zeros(new Shape(numOutputs), DataType.FLOAT32, device);

        // Attach gradients
        NDList params =
                new NDList(
                        W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq,
                        b_q);
        for (NDArray param : params) {
            param.setRequiresGradient(true);
        }
        return params;
    }

    public static NDArray normal(Shape shape, Device device) {
        return manager.randomNormal(0, 0.01f, shape, DataType.FLOAT32, device);
    }

    public static NDList three(int numInputs, int numHiddens, Device device) {
        return new NDList(
                normal(new Shape(numInputs, numHiddens), device),
                normal(new Shape(numHiddens, numHiddens), device),
                manager.zeros(new Shape(numHiddens), DataType.FLOAT32, device));
    }
}
