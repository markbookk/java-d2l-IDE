import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.*;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.*;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.components.Figure;

import java.io.IOException;
import java.util.*;
import java.util.stream.Stream;

public class Main {

    public static NDManager manager;

    public static void main(String[] args) throws IOException, TranslateException {
        manager = NDManager.newBaseManager(Functions.tryGpu(0));

        Seq2SeqEncoder encoder = new Seq2SeqEncoder(10, 8, 16, 2, 0f);
        Seq2SeqAttentionDecoder decoder = new Seq2SeqAttentionDecoder(10, 8, 16, 2, 0f);

        NDArray X = manager.zeros(new Shape(4, 7));
        NDList temp = encoder.forward(new ParameterStore(manager, false), new NDList(X), false);
        temp.add(manager.create(new Shape(0)));
        NDList state = decoder.beginState(new NDList(temp));

        NDList decoderOutput =
                decoder.forward(
                        new ParameterStore(manager, false), new NDList(X).addAll(state), false);
        NDArray output = decoderOutput.get(0);
        state = decoderOutput.subNDList(1);
        System.out.println(output.getShape());
        System.out.println(state.size());
        System.out.println(state.get(0).getShape());
        temp = new NDList(state.subList(1, state.getShapes().length - 1));
        System.out.println(temp.size());
        System.out.println(temp.get(0).getShape());

        int embedSize = 32;
        int numHiddens = 32;
        int numLayers = 2;
        float dropout = 0.1f;
        int batchSize = 64;
        int numSteps = 10;
        float lr = 0.005f;
        int numEpochs = 250;
        Device device = Functions.tryGpu(0);

        Pair<ArrayDataset, Pair<Vocab, Vocab>> dataNMT =
                NMT.loadDataNMT(batchSize, numSteps, 600, manager);
        ArrayDataset dataset = dataNMT.getKey();
        Vocab srcVocab = dataNMT.getValue().getKey();
        Vocab tgtVocab = dataNMT.getValue().getValue();

        encoder = new Seq2SeqEncoder(srcVocab.length(), embedSize, numHiddens, numLayers, dropout);
        decoder =
                new Seq2SeqAttentionDecoder(
                        tgtVocab.length(), embedSize, numHiddens, numLayers, dropout);

        EncoderDecoder net = new EncoderDecoder(encoder, decoder);
        trainSeq2Seq(net, dataset, lr, numEpochs, tgtVocab, device);

        String[] engs = {"go .", "i lost .", "he\'s calm .", "i\'m home ."};
        String[] fras = {"va !", "j\'ai perdu .", "il est calme .", "je suis chez moi ."};
        ArrayList<NDArray> decAttentionWeightSeq = new ArrayList<>();
        for (int i = 0; i < engs.length; i++) {
            Pair<String, ArrayList<NDArray>> pair =
                    predictSeq2Seq(net, engs[i], srcVocab, tgtVocab, numSteps, device, true);
            String translation = pair.getKey();
            decAttentionWeightSeq = pair.getValue();
            System.out.format(
                    "%s => %s, bleu %.3f\n", engs[i], translation, bleu(translation, fras[i], 2));
        }

        NDList steps = new NDList();
        for (NDArray step : decAttentionWeightSeq) {
            steps.add(step.get(0).get(0).get(0).reshape(1));
        }
        NDArray attentionWeights = NDArrays.concat(steps, 0).reshape(1, -1, 1, numSteps);
        // Plus one to include the end-of-sequence token
        Figure fig =
                PlotUtils.showHeatmaps(
                        attentionWeights.get(
                                new NDIndex(
                                        ":, :, :, :{}",
                                        engs[engs.length - 1].split(" ").length + 1)),
                        "Key positions",
                        "Query positions",
                        new String[] {""},
                        500,
                        700);
        Plot.show(fig);
    }

    public static void trainSeq2Seq(
            EncoderDecoder net,
            ArrayDataset dataset,
            float lr,
            int numEpochs,
            Vocab tgtVocab,
            Device device)
            throws IOException, TranslateException {
        Loss loss = new MaskedSoftmaxCELoss();
        Tracker lrt = Tracker.fixed(lr);
        Optimizer adam = Optimizer.adam().optLearningRateTracker(lrt).build();

        DefaultTrainingConfig config =
                new DefaultTrainingConfig(loss)
                        .optOptimizer(adam) // Optimizer (loss function)
                        //                        .optInitializer(Initializer.ZEROS,
                        // "")//XavierInitializer(), "")
                        .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

        Model model = Model.newInstance("");
        model.setBlock(net);
        Trainer trainer = model.newTrainer(config);

        //        Animator animator = new Animator();
        StopWatch watch;
        Accumulator metric;
        double lossValue = 0, speed = 0;
        for (int epoch = 1; epoch <= numEpochs; epoch++) {
            watch = new StopWatch();
            metric = new Accumulator(2); // Sum of training loss, no. of tokens
            try (NDManager childManager = manager.newSubManager(device)) {
                // Iterate over dataset
                for (Batch batch : dataset.getData(childManager)) {
                    NDArray X = batch.getData().get(0);
                    NDArray lenX = batch.getData().get(1);
                    NDArray Y = batch.getLabels().get(0);
                    NDArray lenY = batch.getLabels().get(1);

                    NDArray bos =
                            childManager
                                    .full(new Shape(Y.getShape().get(0)), tgtVocab.getIdx("<bos>"))
                                    .reshape(-1, 1);
                    NDArray decInput =
                            NDArrays.concat(
                                    new NDList(bos, Y.get(new NDIndex(":, :-1"))),
                                    1); // Teacher forcing
                    try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                        NDArray yHat =
                                net.forward(
                                                new ParameterStore(manager, false),
                                                new NDList(X, decInput, lenX),
                                                true)
                                        .get(0);
                        NDArray l = loss.evaluate(new NDList(Y, lenY), new NDList(yHat));
                        gc.backward(l);
                        metric.add(new float[] {l.sum().getFloat(), lenY.sum().getLong()});
                    }
                    Training.gradClipping(net, 1, childManager);
                    // Update parameters
                    trainer.step();
                }
            }
            lossValue = metric.get(0) / metric.get(1);
            speed = metric.get(1) / watch.stop();
            if ((epoch + 1) % 10 == 0) {
                //                    animator.add(epoch + 1, (float) ppl, "ppl");
                //                    animator.show();
            }
            System.out.format(
                    "epoch: %d, loss: %.3f, %.1f tokens/sec on %s%n",
                    epoch, lossValue, speed, device.toString());

            trainer.notifyListeners(listener -> listener.onEpoch(trainer));
        }
        System.out.format(
                "loss: %.3f, %.1f tokens/sec on %s%n", lossValue, speed, device.toString());
    }

    public static Pair<String, ArrayList<NDArray>> predictSeq2Seq(
            EncoderDecoder net,
            String srcSentence,
            Vocab srcVocab,
            Vocab tgtVocab,
            int numSteps,
            Device device,
            boolean saveAttentionWeights)
            throws IOException, TranslateException {
        Integer[] srcTokens =
                Stream.concat(
                                Arrays.stream(
                                        srcVocab.getIdxs(srcSentence.toLowerCase().split(" "))),
                                Arrays.stream(new Integer[] {srcVocab.getIdx("<eos>")}))
                        .toArray(Integer[]::new);
        NDArray encValidLen = manager.create(srcTokens.length);
        int[] truncateSrcTokens = NMT.truncatePad(srcTokens, numSteps, srcVocab.getIdx("<pad>"));
        // Add the batch axis
        NDArray encX = manager.create(truncateSrcTokens).expandDims(0);
        NDList encOutputs =
                net.encoder.forward(
                        new ParameterStore(manager, false), new NDList(encX, encValidLen), false);
        NDList decState = net.decoder.beginState(encOutputs.addAll(new NDList(encValidLen)));
        // Add the batch axis
        NDArray decX = manager.create(new float[] {tgtVocab.getIdx("<bos>")}).expandDims(0);
        ArrayList<Integer> outputSeq = new ArrayList<>();
        ArrayList<NDArray> attentionWeightSeq = new ArrayList<>();
        for (int i = 0; i < numSteps; i++) {
            NDList output =
                    net.decoder.forward(
                            new ParameterStore(manager, false),
                            new NDList(decX).addAll(decState),
                            false);
            NDArray Y = output.get(0);
            decState = output.subNDList(1);
            // We use the token with the highest prediction likelihood as the input
            // of the decoder at the next time step
            decX = Y.argMax(2);
            int pred = (int) decX.squeeze(0).getLong();
            // Save attention weights (to be covered later)
            if (saveAttentionWeights) {
                attentionWeightSeq.add(net.decoder.attentionWeights.get(0));
            }
            // Once the end-of-sequence token is predicted, the generation of the
            // output sequence is complete
            if (pred == tgtVocab.getIdx("<eos>")) {
                break;
            }
            outputSeq.add(pred);
        }
        String outputString =
                String.join(" ", tgtVocab.toTokens(outputSeq).toArray(new String[] {}));
        return new Pair<>(outputString, attentionWeightSeq);
    }

    public static double bleu(String predSeq, String labelSeq, int k) {
        /* Compute the BLEU. */
        String[] predTokens = predSeq.split(" ");
        String[] labelTokens = labelSeq.split(" ");
        int lenPred = predTokens.length, lenLabel = labelTokens.length;
        double score = Math.exp(Math.min(0, 1 - lenLabel / lenPred));
        for (int n = 1; n < k + 1; n++) {
            int numMatches = 0;
            HashMap<String, Integer> labelSubs = new HashMap<>();
            for (int i = 0; i < lenLabel - n + 1; i++) {
                String key =
                        String.join(" ", Arrays.copyOfRange(labelTokens, i, i + n, String[].class));
                labelSubs.put(key, labelSubs.getOrDefault(key, 0) + 1);
            }
            for (int i = 0; i < lenPred - n + 1; i++) {
                String key =
                        String.join(" ", Arrays.copyOfRange(predTokens, i, i + n, String[].class));
                if (labelSubs.getOrDefault(key, 0) > 0) {
                    numMatches += 1;
                    labelSubs.put(key, labelSubs.getOrDefault(key, 0) - 1);
                }
            }
            score *= Math.pow(numMatches / (lenPred - n + 1), Math.pow(0.5, n));
        }
        return score;
    }
}
