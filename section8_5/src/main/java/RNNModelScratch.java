import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.util.Pair;

/** An RNN Model implemented from scratch. */
public class RNNModelScratch {
    public int vocabSize;
    public int numHiddens;
    public NDList params;
    public Main.TriFunction<Integer, Integer, Device, NDArray> initState;
    public Main.TriFunction<NDArray, NDArray, NDList, Pair> forwardFn;

    public RNNModelScratch(
            int vocabSize,
            int numHiddens,
            Device device,
            Main.TriFunction<Integer, Integer, Device, NDList> getParams,
            Main.TriFunction<Integer, Integer, Device, NDArray> initRNNState,
            Main.TriFunction<NDArray, NDArray, NDList, Pair> forwardFn) {
        this.vocabSize = vocabSize;
        this.numHiddens = numHiddens;
        this.params = getParams.apply(vocabSize, numHiddens, device);
        this.initState = initRNNState;
        this.forwardFn = forwardFn;
    }

    public Pair forward(NDArray X, NDArray state) {
        X = X.transpose().oneHot(this.vocabSize);
        return this.forwardFn.apply(X, state, this.params);
    }

    public NDArray beginState(int batchSize, Device device) {
        return this.initState.apply(batchSize, this.numHiddens, device);
    }
}
