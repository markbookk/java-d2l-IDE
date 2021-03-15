import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.loss.Loss;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
import ai.djl.util.Pair;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class Main {
    public static NDManager manager;

    public static void main(String[] args) throws Exception {
        manager = NDManager.newBaseManager();
        int batchSize = 32;
        int numSteps = 35;
        Pair<ArrayList<NDList>, Vocab> timeMachine =
                loadDataTimeMachine(batchSize, numSteps, false, 10000);
        List<NDList> trainIter = timeMachine.getKey();
        Vocab vocab = timeMachine.getValue();

        int numHiddens = 512;
        TriFunction<Integer, Integer, Device, NDList> getParamsFn = (a, b, c) -> getParams(a, b, c);
        TriFunction<Integer, Integer, Device, NDArray> initRNNStateFn =
                (a, b, c) -> initRNNState(a, b, c);
        TriFunction<NDArray, NDArray, NDList, Pair> rnnFn = (a, b, c) -> rnn(a, b, c);

        NDArray X = manager.arange(10).reshape(new Shape(2, 5));

        RNNModelScratch net =
                new RNNModelScratch(
                        vocab.length(), numHiddens, tryGpu(0), getParamsFn, initRNNStateFn, rnnFn);
        NDArray state = net.beginState((int) X.getShape().getShape()[0], tryGpu(0));
        Pair<NDArray, NDArray> pairResult = net.forward(X.toDevice(tryGpu(0), false), state);
        NDArray Y = pairResult.getKey();
        NDArray newState = pairResult.getValue();
        System.out.println(Y.getShape());
        System.out.println(newState.getShape());

        predictCh8("time traveller ", 10, net, vocab, tryGpu(0));
        int numEpochs = 500;
        int lr = 1;
        trainCh8(net, trainIter, vocab, lr, numEpochs, tryGpu(0), false);
    }

    /** Train a model. */
    public static void trainCh8(
            RNNModelScratch net,
            List<NDList> trainIter,
            Vocab vocab,
            int lr,
            int numEpochs,
            Device device,
            boolean useRandomIter) {
        SoftmaxCrossEntropyLoss loss = new SoftmaxCrossEntropyLoss();
        //            Animator animator = new Animator();
        // Initialize
        voidTwoFunction<Integer, NDManager> updater =
                (batchSize, subManager) -> Training.sgd(net.params, lr, batchSize, subManager);
        Function<String, String> predict = (prefix) -> predictCh8(prefix, 50, net, vocab, device);
        // Train and predict
        double ppl = 0.0;
        double speed = 0.0;
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            Pair<Double, Double> pair =
                    trainEpochCh8(net, trainIter, loss, updater, device, useRandomIter);
            ppl = pair.getKey();
            speed = pair.getValue();
            if ((epoch + 1) % 10 == 0) {
                //                    animator.add(epoch + 1, (float) ppl, "");
            }
            System.out.format(
                    "epochs: %d, perplexity: %.1f, %.1f tokens/sec on %s%n", epoch, ppl, speed, device.toString());
        }
        System.out.format(
                "perplexity: %.1f, %.1f tokens/sec on %s%n", ppl, speed, device.toString());
        System.out.println(predict.apply("time traveller"));
        System.out.println(predict.apply("traveller"));
    }

    /** Train a model within one epoch. */
    public static Pair<Double, Double> trainEpochCh8(
            RNNModelScratch net,
            List<NDList> trainIter,
            Loss loss,
            voidTwoFunction<Integer, NDManager> updater,
            Device device,
            boolean useRandomIter) {
        StopWatch watch = new StopWatch();
        watch.start();
        Accumulator metric = new Accumulator(2); // Sum of training loss, no. of tokens
        try (NDManager childManager = manager.newSubManager()) {
            NDArray state = null;
            for (NDList pair : trainIter) {
                NDArray X = pair.get(0).toDevice(Functions.tryGpu(0), true);
                X.attach(childManager);
                NDArray Y = pair.get(1).toDevice(Functions.tryGpu(0), true);
                Y.attach(childManager);
                if (state == null || useRandomIter) {
                    // Initialize `state` when either it is the first iteration or
                    // using random sampling
                    state = net.beginState((int) X.getShape().getShape()[0], device);
                } else {
                    state.stopGradient();
                }
                state.attach(childManager);

                NDArray y = Y.transpose().reshape(new Shape(-1));
                X = X.toDevice(device, false);
                y = y.toDevice(device, false);
                try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                    Pair<NDArray, NDArray> pairResult = net.forward(X, state);
                    NDArray yHat = pairResult.getKey();
                    state = pairResult.getValue();
                    NDArray l = loss.evaluate(new NDList(y), new NDList(yHat)).mean();
                    gc.backward(l);
                    metric.add(new float[] {l.getFloat() * y.size(), y.size()});
                }
                gradClipping(net, 1, childManager);
                updater.apply(1, childManager); // Since the `mean` function has been invoked
            }
        }
        return new Pair<>(Math.exp(metric.get(0) / metric.get(1)), metric.get(1) / watch.stop());
    }

    /** Clip the gradient. */
    public static void gradClipping(RNNModelScratch net, int theta, NDManager manager) {
        double result = 0;
        for (NDArray p : net.params) {
            NDArray gradient = p.getGradient();
            gradient.attach(manager);
            result += gradient.pow(2).sum().getFloat();
        }
        double norm = Math.sqrt(result);
        if (norm > theta) {
            for (NDArray param : net.params) {
                NDArray gradient = param.getGradient();
                gradient.muli(theta / norm);
            }
        }
    }

    /** Generate new characters following the `prefix`. */
    public static String predictCh8(
            String prefix, int numPreds, RNNModelScratch net, Vocab vocab, Device device) {
        NDArray state = net.beginState(1, device);
        List<Integer> outputs = new ArrayList<>();
        outputs.add(vocab.getIdx("" + prefix.charAt(0)));
        SimpleFunction<NDArray> getInput =
                () ->
                        manager.create(outputs.get(outputs.size() - 1))
                                .toDevice(device, false)
                                .reshape(new Shape(1, 1));
        for (char c : prefix.substring(1).toCharArray()) { // Warm-up period
            state = (NDArray) net.forward(getInput.apply(), state).getValue();
            outputs.add(vocab.getIdx("" + c));
        }

        NDArray y;
        for (int i = 0; i < numPreds; i++) {
            Pair<NDArray, NDArray> pair = net.forward(getInput.apply(), state);
            y = pair.getKey();
            state = pair.getValue();

            outputs.add((int) y.argMax(1).reshape(new Shape(1)).getLong(0L));
        }
        StringBuilder output = new StringBuilder();
        for (int i : outputs) {
            output.append(vocab.idxToToken.get(i));
        }
        return output.toString();
    }

    /** Return the i'th GPU if it exists, otherwise return the CPU */
    public static Device tryGpu(int i) {
        return Device.getGpuCount() >= i + 1 ? Device.gpu(i) : Device.cpu();
    }

    public static NDArray initRNNState(int batchSize, int numHiddens, Device device) {
        return manager.zeros(new Shape(batchSize, numHiddens), DataType.FLOAT32, device);
    }

    public static NDList getParams(int vocabSize, int numHiddens, Device device) {
        int numOutputs = vocabSize;
        int numInputs = vocabSize;

        // Hidden layer parameters
        NDArray W_xh = normal(new Shape(numInputs, numHiddens), device);
        NDArray W_hh = normal(new Shape(numHiddens, numHiddens), device);
        NDArray b_h = manager.zeros(new Shape(numHiddens), DataType.FLOAT32, device);
        // Output layer parameters
        NDArray W_hq = normal(new Shape(numHiddens, numOutputs), device);
        NDArray b_q = manager.zeros(new Shape(numOutputs), DataType.FLOAT32, device);

        // Attach gradients
        NDList params = new NDList(W_xh, W_hh, b_h, W_hq, b_q);
        for (NDArray param : params) {
            param.attachGradient();
        }
        return params;
    }

    public static Pair rnn(NDArray inputs, NDArray state, NDList params) {
        // Shape of `inputs`: (`numSteps`, `batchSize`, `vocabSize`)
        NDArray W_xh = params.get(0);
        NDArray W_hh = params.get(1);
        NDArray b_h = params.get(2);
        NDArray W_hq = params.get(3);
        NDArray b_q = params.get(4);
        NDArray H = state;

        NDList outputs = new NDList();
        // Shape of `X`: (`batchSize`, `vocabSize`)
        NDArray X, Y;
        for (int i = 0; i < inputs.size(0); i++) {
            X = inputs.get(i);
            H = (X.dot(W_xh).add(H.dot(W_hh)).add(b_h)).tanh();
            Y = H.dot(W_hq).add(b_q);
            outputs.add(Y);
        }
        return new Pair(outputs.size() > 1 ? NDArrays.concat(outputs) : outputs.get(0), H);
    }

    public static NDArray normal(Shape shape, Device device) {
        return manager.randomNormal(0f, 0.01f, shape, DataType.FLOAT32, device);
    }

    @FunctionalInterface
    public interface TriFunction<T, U, V, W> {
        public W apply(T t, U u, V v);
    }

    @FunctionalInterface
    public interface QuadFunction<T, U, V, W, R> {
        public R apply(T t, U u, V v, W w);
    }

    @FunctionalInterface
    public interface SimpleFunction<T> {
        public T apply();
    }

    @FunctionalInterface
    public interface voidFunction<T> {
        public void apply(T t);
    }

    @FunctionalInterface
    public interface voidTwoFunction<T, U> {
        public void apply(T t, U u);
    }

    /** Return the iterator and the vocabulary of the time machine dataset. */
    public static Pair<ArrayList<NDList>, Vocab> loadDataTimeMachine(
            int batchSize, int numSteps, boolean useRandomIter, int maxTokens)
            throws IOException, Exception {

        SeqDataLoader seqData =
                new SeqDataLoader(batchSize, numSteps, useRandomIter, maxTokens, manager);
        return new Pair(seqData.dataIter, seqData.vocab); // ArrayList<NDList>, Vocab
    }
}
