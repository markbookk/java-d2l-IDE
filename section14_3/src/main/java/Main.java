import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.*;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.training.*;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import ai.djl.util.ZipUtils;
import tech.tablesaw.plotly.components.*;
import tech.tablesaw.plotly.traces.HistogramTrace;

import java.io.*;
import java.net.URL;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.*;

public class Main {

    public static NDManager manager;

    public static void main(String[] args) throws IOException, TranslateException {
        manager = NDManager.newBaseManager(Functions.tryGpu(0));
        String[][] sentences = readPTB();
        System.out.println("# sentences: " + sentences.length);

        Vocab vocab = new Vocab(sentences, 10, new String[] {});
        System.out.println(vocab.length());

        String[][] subsampled = subSampling(sentences, vocab);

        double[] y1 = new double[sentences.length];
        for (int i = 0; i < sentences.length; i++) y1[i] = sentences[i].length;
        double[] y2 = new double[subsampled.length];
        for (int i = 0; i < subsampled.length; i++) y2[i] = subsampled[i].length;

        HistogramTrace trace1 =
                HistogramTrace.builder(y1).opacity(.75).name("origin").nBinsX(20).build();
        HistogramTrace trace2 =
                HistogramTrace.builder(y2).opacity(.75).name("subsampled").nBinsX(20).build();

        Layout layout =
                Layout.builder()
                        .barMode(Layout.BarMode.GROUP)
                        .showLegend(true)
                        .xAxis(Axis.builder().title("# tokens per sentence").build())
                        .yAxis(Axis.builder().title("count").build())
                        .build();
        //        Plot.show(new Figure(layout, trace1, trace2));

        System.out.println(compareCounts("the", sentences, subsampled));

        System.out.println(compareCounts("join", sentences, subsampled));

        Integer[][] corpus = new Integer[subsampled.length][];
        for (int i = 0; i < subsampled.length; i++) {
            corpus[i] = vocab.getIdxs(subsampled[i]);
        }

        Integer[][] tinyDataset =
                new Integer[][] {
                    IntStream.range(0, 7)
                            .boxed()
                            .collect(Collectors.toList())
                            .toArray(new Integer[] {}),
                    IntStream.range(7, 10)
                            .boxed()
                            .collect(Collectors.toList())
                            .toArray(new Integer[] {})
                };

        System.out.println("dataset " + Arrays.deepToString(tinyDataset));

        Pair<ArrayList<Integer>, ArrayList<ArrayList<Integer>>> centerContextPair =
                getCentersAndContext(tinyDataset, 2);
        for (int i = 0; i < centerContextPair.getValue().size(); i++) {
            System.out.println(
                    "Center "
                            + centerContextPair.getKey().get(i)
                            + " has contexts"
                            + centerContextPair.getValue().get(i));
        }

        centerContextPair = getCentersAndContext(corpus, 5);
        ArrayList<Integer> allCenters = centerContextPair.getKey();
        ArrayList<ArrayList<Integer>> allContexts = centerContextPair.getValue();
        System.out.println("# center-context pairs:" + allCenters.size());

        RandomGenerator generator =
                new RandomGenerator(Arrays.asList(new Double[] {2.0, 3.0, 4.0}));
        Integer[] generatorOutput = new Integer[10];
        for (int i = 0; i < 10; i++) {
            generatorOutput[i] = generator.draw();
        }

        System.out.println(Arrays.toString(generatorOutput));

        ArrayList<ArrayList<Integer>> allNegatives = getNegatives(allContexts, corpus, 5);

        ArrayList<Object> x1 = new ArrayList<>();
        x1.add(1);
        ArrayList<Integer> temp = new ArrayList<>();
        Collections.addAll(temp, new Integer[]{2, 2});
        x1.add(temp);
        Collections.addAll(temp, new Integer[]{3, 3, 3, 3});
        x1.add(temp);

        ArrayList<Object> x2 = new ArrayList<>();
        x2.add(1);
        temp = new ArrayList<>();
        Collections.addAll(temp, new Integer[]{2, 2, 2});
        x2.add(temp);
        Collections.addAll(temp, new Integer[]{3, 3});
        x2.add(temp);

        ArrayList<ArrayList<Object>> data = new ArrayList<>();
        data.add(x1);
        data.add(x2);
        NDList batch = batchify(data, manager);

    }

    public static NDList batchify(ArrayList<ArrayList<Object>> data, NDManager manager) {
        int maxLen = 0;
        for (ArrayList<Object> obj : data) {
            ArrayList<Integer> c = (ArrayList<Integer>) obj.get(1);
            ArrayList<Integer> n = (ArrayList<Integer>) obj.get(2);
            maxLen = Math.max(maxLen, c.size() + n.size());
        }
        ArrayList<Integer> centers = new ArrayList<>();
        ArrayList<ArrayList<Integer>> contextNegatives = new ArrayList<>();
        ArrayList<ArrayList<Integer>> masks = new ArrayList<>();
        ArrayList<ArrayList<Integer>> labels = new ArrayList<>();

        for (ArrayList<Object> obj : data) {
            int center = (int) obj.get(0);
            ArrayList<Integer> context = (ArrayList<Integer>) obj.get(1);
            ArrayList<Integer> negative = (ArrayList<Integer>) obj.get(1);
            int curLen = context.size() + negative.size();
            centers.add(center);

            context.addAll(negative);
            for (int i = 0; i < maxLen - curLen; i++) {
                context.add(0);
            }
            contextNegatives.add(centers);

            ArrayList<Integer> temp = new ArrayList<>();
            for (int i = 0; i < curLen; i++) {
                temp.add(1);
            }
            for (int i = 0; i < maxLen - curLen; i++) {
                temp.add(0);
            }
            masks.add(temp);

            temp = new ArrayList<>();
            for (int i = 0; i < context.size(); i++) {
                temp.add(1);
            }
            for (int i = 0; i < maxLen - context.size(); i++) {
                temp.add(0);
            }
            masks.add(temp);
        }
        return new NDList(
                manager.create(centers.stream().mapToInt(i -> i).toArray())
                        .reshape(new Shape(-1, 1)),
                manager.create(
                        contextNegatives.stream()
                                .map(u -> u.stream().mapToInt(i -> i).toArray())
                                .toArray(int[][]::new)),
                manager.create(
                        masks.stream()
                                .map(u -> u.stream().mapToInt(i -> i).toArray())
                                .toArray(int[][]::new)),
                manager.create(
                        labels.stream()
                                .map(u -> u.stream().mapToInt(i -> i).toArray())
                                .toArray(int[][]::new)));
    }

    public static ArrayList<ArrayList<Integer>> getNegatives(
            ArrayList<ArrayList<Integer>> allContexts, Integer[][] corpus, int K) {
        LinkedHashMap<Object, Integer> counter = Vocab.countCorpus2D(corpus);
        ArrayList<Double> samplingWeights = new ArrayList<>();
        for (int i = 0; i < counter.size(); i++) {
            samplingWeights.add(Math.pow(counter.get(i), .75));
        }
        ArrayList<ArrayList<Integer>> allNegatives = new ArrayList<>();
        RandomGenerator generator = new RandomGenerator(samplingWeights);
        for (ArrayList<Integer> contexts : allContexts) {
            ArrayList<Integer> negatives = new ArrayList<>();
            while (negatives.size() < contexts.size() * K) {
                Integer neg = generator.draw();
                // Noise words cannot be context words
                if (!contexts.contains(neg)) {
                    negatives.add(neg);
                }
                allNegatives.add(negatives);
            }
        }
        return allNegatives;
    }

    public static Pair<ArrayList<Integer>, ArrayList<ArrayList<Integer>>> getCentersAndContext(
            Integer[][] corpus, int maxWindowSize) {
        ArrayList<Integer> centers = new ArrayList<>();
        ArrayList<ArrayList<Integer>> contexts = new ArrayList<>();

        for (Integer[] line : corpus) {
            // Each sentence needs at least 2 words to form a "central target word
            // - context word" pair
            if (line.length < 2) {
                continue;
            }
            centers.addAll(Arrays.asList(line));
            for (int i = 0; i < line.length; i++) { // Context window centered at i
                int windowSize = new Random().nextInt(maxWindowSize - 1) + 1;
                List<Integer> indices =
                        IntStream.range(
                                        Math.max(0, i - windowSize),
                                        Math.min(line.length, i + 1 + windowSize))
                                .boxed()
                                .collect(Collectors.toList());
                // Exclude the central target word from the context words
                indices.remove(indices.indexOf(i));
                ArrayList<Integer> temp = new ArrayList<>();
                for (Integer idx : indices) {
                    temp.add(line[idx]);
                }
                contexts.add(temp);
            }
        }
        return new Pair<>(centers, contexts);
    }

    public static String[][] readPTB() throws IOException {
        String ptbURL = "http://d2l-data.s3-accelerate.amazonaws.com/ptb.zip";
        InputStream input = new URL(ptbURL).openStream();
        ZipUtils.unzip(input, Paths.get("./"));

        ArrayList<String> lines = new ArrayList<>();
        File file = new File("./ptb/ptb.train.txt");
        Scanner myReader = new Scanner(file);
        while (myReader.hasNextLine()) {
            lines.add(myReader.nextLine());
        }
        String[][] tokens = new String[lines.size()][];
        for (int i = 0; i < lines.size(); i++) {
            tokens[i] = lines.get(i).trim().split(" ");
        }
        return tokens;
    }

    public static String[][] subSampling(String[][] sentences, Vocab vocab) {
        for (int i = 0; i < sentences.length; i++) {
            for (int j = 0; j < sentences[i].length; j++) {
                sentences[i][j] = vocab.idxToToken.get(vocab.getIdx(sentences[i][j]));
            }
        }
        // Count the frequency for each word
        LinkedHashMap<Object, Integer> counter = vocab.countCorpus2D(sentences);
        int numTokens = 0;
        for (Integer value : counter.values()) {
            numTokens += value;
        }

        // Now do the subsampling
        String[][] output = new String[sentences.length][];
        for (int i = 0; i < sentences.length; i++) {
            ArrayList<String> tks = new ArrayList<>();
            for (int j = 0; j < sentences[i].length; j++) {
                String tk = sentences[i][j];
                if (keep(sentences[i][j], counter, numTokens)) {
                    tks.add(tk);
                }
            }
            output[i] = tks.toArray(new String[tks.size()]);
        }

        return output;
    }

    public static boolean keep(
            String token, LinkedHashMap<Object, Integer> counter, int numTokens) {
        // Return True if to keep this token during subsampling
        //        return new Random().nextFloat() < Math.sqrt(1e-4 / counter.get(token) *
        // numTokens);
        return 0.5f < Math.sqrt(1e-4 / counter.get(token) * numTokens);
    }

    public static String compareCounts(String token, String[][] sentences, String[][] subsampled) {
        int beforeCount = 0;
        for (int i = 0; i < sentences.length; i++) {
            for (int j = 0; j < sentences[i].length; j++) {
                if (sentences[i][j].equals(token)) beforeCount += 1;
            }
        }

        int afterCount = 0;
        for (int i = 0; i < subsampled.length; i++) {
            for (int j = 0; j < subsampled[i].length; j++) {
                if (subsampled[i][j].equals(token)) afterCount += 1;
            }
        }

        return "# of \"the\": before=" + beforeCount + ", after=" + afterCount;
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
                        //                                                .optInitializer(new
                        // XavierInitializer(), "")
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
                        gradClipping(net, 1, manager);
                        trainer.step();
                        yHat =
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

    public static void gradClipping(Object net, int theta, NDManager manager) {
        double result = 0;
        NDList params;
        params = new NDList();
        for (Pair<String, Parameter> pair : ((AbstractBlock) net).getParameters()) {
            params.add(pair.getValue().getArray());
        }
        for (NDArray p : params) {
            NDArray gradient = p.getGradient().stopGradient();
            gradient.attach(manager);
            result += gradient.pow(2).sum().getFloat();
        }
        double norm = Math.sqrt(result);
        //        if (norm > theta) {
        for (NDArray param : params) {
            NDArray gradient = param.getGradient();
            //                gradient.muli(theta / norm);
            gradient.addi(2);
            System.out.println("");
        }
        //        }
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
