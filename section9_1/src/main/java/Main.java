import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
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

        Functions.TriFunction<Integer, Integer, Device, NDList> getParamsFn = (a, b, c) -> getParams(a, b, c);
        Functions.TriFunction<Integer, Integer, Device, NDList> initGruStateFn =
                (a, b, c) -> initGruState(a, b, c);
        Functions.TriFunction<NDArray, NDArray, NDList, Pair> gruFn = (a, b, c) -> gru(a, b, c);

        RNNModelScratch model =
                new RNNModelScratch(vocabSize, numHiddens, device,
                        getParamsFn, initGruStateFn, gruFn);
        TimeMachine.trainCh8(model, dataset, vocab, lr, numEpochs, device, false, manager);

    }

    public static NDList initGruState(int batchSize, int numHiddens, Device device) {
        return new NDList(manager.zeros(new Shape(batchSize, numHiddens), DataType.FLOAT32, device));
    }

    public static Pair<NDArray, NDArray> gru(NDArray inputs, NDArray state, NDList params) {
        NDArray W_xz = params.get(0);
        NDArray W_hz = params.get(1);
        NDArray b_z = params.get(2);

        NDArray W_xr = params.get(3);
        NDArray W_hr = params.get(4);
        NDArray b_r = params.get(5);

        NDArray W_xh = params.get(6);
        NDArray W_hh = params.get(7);
        NDArray b_h = params.get(8);

        NDArray W_hq = params.get(9);
        NDArray b_q  = params.get(10);

        NDArray H = state;
        NDList outputs = new NDList();
        NDArray X, Y;
        for (int i = 0; i < inputs.size(0); i++) {
            X = inputs.get(i);
            NDArray Z = Activation.sigmoid(X.dot(W_xz).add(H.dot(W_hz).add(b_z)));
            NDArray R = Activation.sigmoid(X.dot(W_xr).add(H.dot(W_hr).add(b_r)));
            NDArray H_tilda = Activation.tanh(X.dot(W_xh).add(R.mul(H).dot(W_hh).add(b_h)));
            H = Z.mul(H).add(Z.mul(-1).add(1).mul(H_tilda));
            Y = H.dot(W_hq).add(b_q);
            outputs.add(Y);
        }
        return new Pair(outputs.size() > 1 ? NDArrays.concat(outputs) : outputs.get(0), H);
    }

    public static NDList getParams(int vocabSize, int numHiddens, Device device) {
        int numInputs = vocabSize;
        int numOutputs = vocabSize;

        // Update gate parameters
        NDList temp = three(numInputs, numHiddens, device);
        NDArray W_xz = temp.get(0);
        NDArray W_hz = temp.get(1);
        NDArray b_z = temp.get(2);

        // Reset gate parameters
        temp = three(numInputs, numHiddens, device);
        NDArray W_xr = temp.get(0);
        NDArray W_hr = temp.get(1);
        NDArray b_r = temp.get(2);

        // Candidate hidden state parameters
        temp = three(numInputs, numHiddens, device);
        NDArray W_xh = temp.get(0);
        NDArray W_hh = temp.get(1);
        NDArray b_h = temp.get(2);

        // Output layer parameters
        NDArray W_hq = normal(new Shape(numHiddens, numOutputs), device);
        NDArray b_q = manager.zeros(new Shape(numOutputs), DataType.FLOAT32, device);

        // Attach gradients
        NDList params = new NDList(W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q);
        for (NDArray param : params) {
            param.attachGradient();
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
