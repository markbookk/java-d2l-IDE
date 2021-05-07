import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding;
import ai.djl.ndarray.*;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.*;
import ai.djl.nn.core.Embedding;
import ai.djl.training.*;
import ai.djl.training.dataset.*;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.*;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import ai.djl.util.ZipUtils;
import tech.tablesaw.plotly.components.Axis;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.traces.HistogramTrace;

import java.io.*;
import java.lang.reflect.Array;
import java.net.URL;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.*;

public class Main {

    public static NDManager manager;

    public static void main(String[] args) throws IOException, TranslateException {
        manager = NDManager.newBaseManager(Functions.tryGpu(0));
        int batchSize = 512;
        int maxWindowSize = 5;
        int numNoiseWords = 5;

        Pair<ArrayDataset, Vocab> pair =
                loadDataPTB(batchSize, maxWindowSize, numNoiseWords, manager);
        ArrayDataset dataset = pair.getKey();
        Vocab vocab = pair.getValue();

        TrainableWordEmbedding embed =
                TrainableWordEmbedding.builder().optNumEmbeddings(20).setEmbeddingSize(4).build();
        embed.initialize(manager, DataType.FLOAT32, new Shape(20, 4));
        Parameter param = embed.getParameters().get(0).getValue();
        System.out.println(
                "Parameter " + param.getName() + " | shape " + param.getArray().getShape());

        NDArray X = manager.create(new int[][] {{1, 2, 3}, {4, 5, 6}});
        System.out.println(
                embed.forward(new ParameterStore(manager, false), new NDList(X), false).get(0));

        System.out.println(
                skipGrams(
                                manager.ones(new Shape(2, 1)),
                                manager.ones(new Shape(2, 4)),
                                embed,
                                embed)
                        .getShape());

        Loss loss = new SigmoidBinaryCrossEntropyLoss();
        NDArray pred = manager.full(new Shape(2, 4), .5f);
        NDArray label = manager.create(new float[][] {{1, 0, 1, 0}, {1, 0, 1, 0}});
        NDArray mask = manager.create(new float[][] {{1, 1, 1, 1}, {1, 1, 0, 0}});
        System.out.println(loss.evaluate(new NDList(label), new NDList(pred.mul(mask))));

        System.out.println(
                loss.evaluate(new NDList(label), new NDList(pred.mul(mask)))
                        .div(mask.sum(new int[] {1}))
                        .mul(mask.getShape().get(1)));

        int embedSize = 100;
        SequentialBlock net = new SequentialBlock();
        net.addAll(
                TrainableWordEmbedding.builder().optNumEmbeddings(20).setEmbeddingSize(4).build(),
                TrainableWordEmbedding.builder().optNumEmbeddings(20).setEmbeddingSize(4).build());

        float lr = 0.01f;
        int numEpochs = 5;
        train(net, dataset, lr, numEpochs, Functions.tryGpu(0), loss);

        System.out.println("");
    }

    public static void train(
            Block net,
            RandomAccessDataset dataset,
            float lr,
            int numEpochs,
            Device device,
            Loss loss)
            throws IOException, TranslateException {
        net.initialize(manager, DataType.FLOAT32, new Shape());
        Model model = Model.newInstance("model");
        model.setBlock(net);

        Tracker lrt = Tracker.fixed(lr);
        Optimizer adam = Optimizer.adam().optLearningRateTracker(lrt).build();

        DefaultTrainingConfig config =
                new DefaultTrainingConfig(loss)
                        .optOptimizer(adam) // Optimizer (loss function)
                        .optDevices(Device.getDevices(1)) // setting the number of GPUs needed
                        .addEvaluator(new Accuracy()) // Model Accuracy
                        .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

        Trainer trainer = model.newTrainer(config);

//        Iterator<Batch> dataIter = dataset.getData(manager).iterator();
//        Iterator<Batch> dataIter = dataset.getData(manager).iterator();
        // Count batches
//        int numBatches = 0;
        //        for (Batch batch : dataIter) {
        //            System.out.println(1);
        //            numBatches += 1;
        //        }
        Accumulator metric = new Accumulator(2);
        StopWatch stopWatch = new StopWatch();
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            stopWatch.start();
            //            for (Batch batch : dataIter) {
            try (NDManager childManager = manager.newSubManager()) {
                Iterator<Batch> dataIter = dataset.getData(childManager).iterator();
                int i = 0;
                while (dataIter.hasNext()) {
                    Batch batch = dataIter.next();
                    NDArray center = batch.getData().get(0);
                    NDArray contextNegative = batch.getData().get(1);
                    NDArray mask = batch.getData().get(2);
                    NDArray label = batch.getData().get(3);

                    NDArray l;
                    try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                        NDArray pred =
                                skipGrams(
                                        center,
                                        contextNegative,
                                        (Embedding) net.getChildren().get(0).getValue(),
                                        (Embedding) net.getChildren().get(1).getValue());

                        l =
                                loss.evaluate(
                                                new NDList(label),
                                                new NDList(
                                                        pred.reshape(label.getShape()).mul(mask)))
                                        .div(mask.sum(new int[] {1}))
                                        .mul(mask.getShape().get(1));
                        gc.backward(l);
                        metric.add(new float[] {l.sum().getFloat(), l.size()});
                    }
                    trainer.step();
                    //                if ((i + 1) % ((int)(numBatches / 5)) == 0 || i == (numBatches
                    // -
                    // 1)) {
                    //                    //animator add!!!
                    //                }
                    System.out.println("Trainer step: " + i);
                    i += 1;
                }
            }
            System.out.format(
                    "epoch: %d, loss: %.3f, %.1f tokens/sec on %s%n",
                    epoch,
                    metric.get(0) / metric.get(1),
                    metric.get(1) / stopWatch.stop(),
                    device.toString());
        }
        System.out.format(
                "loss: %.3f, %.1f tokens/sec on %s%n",
                metric.get(0) / metric.get(1), metric.get(1) / stopWatch.stop(), device.toString());
    }

    public static NDArray skipGrams(
            NDArray center, NDArray contextsAndNegatives, Embedding embedV, Embedding embedU) {
        NDArray v =
                embedV.forward(new ParameterStore(manager, false), new NDList(center), false)
                        .get(0);
        NDArray u =
                embedU.forward(
                                new ParameterStore(manager, false),
                                new NDList(contextsAndNegatives),
                                false)
                        .get(0);
        NDArray pred = v.batchDot(u.swapAxes(1, 2));
        return pred;
    }

    public static Pair<ArrayDataset, Vocab> loadDataPTB(
            int batchSize, int maxWindowSize, int numNoiseWords, NDManager manager)
            throws IOException, TranslateException {
        String[][] sentences = readPTB();
        Vocab vocab = new Vocab(sentences, 10, new String[] {});
        String[][] subSampled = subSampling(sentences, vocab);
        Integer[][] corpus = new Integer[subSampled.length][];
        for (int i = 0; i < subSampled.length; i++) {
            corpus[i] = vocab.getIdxs(subSampled[i]);
        }
        Pair<ArrayList<Integer>, ArrayList<ArrayList<Integer>>> pair =
                getCentersAndContext(corpus, maxWindowSize);
        ArrayList<ArrayList<Integer>> negatives =
                getNegatives(pair.getValue(), corpus, numNoiseWords);

        NDList ndArrays =
                convertNDArray(new Object[] {pair.getKey(), pair.getValue(), negatives}, manager);
        NDManager childManager = manager.newSubManager(Functions.tryGpu(0));
        ArrayDataset dataset =
                new ArrayDataset.Builder()
                        .setData(ndArrays.get(0), ndArrays.get(1), ndArrays.get(2))
                        .optDataBatchifier(
                                new Batchifier() {
                                    @Override
                                    public NDList batchify(NDList[] ndLists) {
                                        return batchifyData(ndLists);
                                    }

                                    @Override
                                    public NDList[] unbatchify(NDList ndList) {
                                        return new NDList[0];
                                    }
                                })
                        .setSampling(batchSize, true)
                        .build();

        return new Pair<>(dataset, vocab);
    }

    public static NDList batchifyData(NDList[] data) {
        NDArray centers = null;
        NDArray contextsNegatives = null;
        NDArray masks = null;
        NDArray labels = null;

        long maxLen = 0;
        for (NDList ndList : data) { // center, context, negative = ndList
            maxLen =
                    Math.max(
                            maxLen,
                            ndList.get(1).countNonzero().getLong()
                                    + ndList.get(2).countNonzero().getLong());
        }
        for (NDList ndList : data) { // center, context, negative = ndList
            NDManager previousManager = ndList.getManager();
            NDManager childManager = manager.newSubManager(Functions.tryGpu(0));
            // Assign temporarily to childManager
            ndList.attach(childManager);

            NDArray center = ndList.get(0);
            NDArray context = ndList.get(1);
            NDArray negative = ndList.get(2);

            NDArray contextNegative = null;
            NDArray mask = null;
            NDArray label = null;
            for (int i = 0; i < context.size(); i++) {
                // If a 0 is found, we want to stop adding these
                // values to NDArray
                if (context.get(i).getInt() == 0) {
                    break;
                }
                contextNegative =
                        contextNegative != null
                                ? contextNegative.concat(context.get(i).reshape(1))
                                : context.get(i).reshape(1);
                mask =
                        mask != null
                                ? mask.concat(childManager.create(1).reshape(1))
                                : childManager.create(1).reshape(1);
                label =
                        label != null
                                ? label.concat(childManager.create(1).reshape(1))
                                : childManager.create(1).reshape(1);
            }
            for (int i = 0; i < negative.size(); i++) {
                // If a 0 is found, we want to stop adding these
                // values to NDArray
                if (negative.get(i).getInt() == 0) {
                    break;
                }
                contextNegative =
                        contextNegative != null
                                ? contextNegative.concat(negative.get(i).reshape(1))
                                : negative.get(i).reshape(1);
                ;
                mask =
                        mask != null
                                ? mask.concat(childManager.create(1).reshape(1))
                                : childManager.create(1).reshape(1);
                label =
                        label != null
                                ? label.concat(childManager.create(0).reshape(1))
                                : childManager.create(0).reshape(1);
            }
            // Fill with zeroes remaining array
            while (contextNegative.size() != maxLen) {
                contextNegative =
                        contextNegative != null
                                ? contextNegative.concat(childManager.create(0).reshape(1))
                                : childManager.create(0).reshape(1);
                mask =
                        mask != null
                                ? mask.concat(childManager.create(0).reshape(1))
                                : childManager.create(0).reshape(1);
                label =
                        label != null
                                ? label.concat(childManager.create(0).reshape(1))
                                : childManager.create(0).reshape(1);
            }

            // Add this NDArrays to output NDArrays
            centers =
                    centers != null
                            ? centers.concat(center.reshape(1, center.size()))
                            : center.reshape(1, center.size());
            contextsNegatives =
                    contextsNegatives != null
                            ? contextsNegatives.concat(
                                    contextNegative.reshape(1, contextNegative.size()))
                            : contextNegative.reshape(1, contextNegative.size());
            masks =
                    masks != null
                            ? masks.concat(mask.reshape(1, mask.size()))
                            : mask.reshape(1, mask.size());
            labels =
                    labels != null
                            ? labels.concat(label.reshape(1, label.size()))
                            : label.reshape(1, label.size());

            // Assign back main manager and clear childManager memory
            new NDList(centers, center, context, negative, contextsNegatives, masks, labels).attach(previousManager);
            childManager.close();
        }

        return new NDList(centers, contextsNegatives, masks, labels);
    }

    public static NDList convertNDArray(Object[] data, NDManager manager) {
        ArrayList<Integer> centers = (ArrayList<Integer>) data[0];
        ArrayList<ArrayList<Integer>> contexts = (ArrayList<ArrayList<Integer>>) data[1];
        ArrayList<ArrayList<Integer>> negatives = (ArrayList<ArrayList<Integer>>) data[2];

        // Create centers NDArray
        NDArray centersNDArray = manager.create(centers.stream().mapToInt(i -> i).toArray());

        // Create contexts NDArray
        int maxLen = 0;
        for (ArrayList<Integer> context : contexts) {
            maxLen = Math.max(maxLen, context.size());
        }
        // Fill arrays with 0s to all have same lengths and be able to create NDArray
        for (ArrayList<Integer> context : contexts) {
            while (context.size() != maxLen) {
                context.add(0);
            }
        }
        NDArray contextsNDArray =
                manager.create(
                        contexts.stream()
                                .map(u -> u.stream().mapToInt(i -> i).toArray())
                                .toArray(int[][]::new));

        // Create negatives NDArray
        maxLen = 0;
        for (ArrayList<Integer> negative : negatives) {
            maxLen = Math.max(maxLen, negative.size());
        }
        // Fill arrays with 0s to all have same lengths and be able to create NDArray
        for (ArrayList<Integer> negative : negatives) {
            while (negative.size() != maxLen) {
                negative.add(0);
            }
        }
        NDArray negativesNDArray =
                manager.create(
                        negatives.stream()
                                .map(u -> u.stream().mapToInt(i -> i).toArray())
                                .toArray(int[][]::new));

        return new NDList(centersNDArray, contextsNDArray, negativesNDArray);
    }

    public static ArrayList<ArrayList<Integer>> getNegatives(
            ArrayList<ArrayList<Integer>> allContexts, Integer[][] corpus, int K) {
        LinkedHashMap<Object, Integer> counter = Vocab.countCorpus2D(corpus);
        ArrayList<Double> samplingWeights = new ArrayList<>();
        for (Map.Entry<Object, Integer> entry : counter.entrySet()) {
            samplingWeights.add(Math.pow(entry.getValue(), .75));
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
            }
            allNegatives.add(negatives);
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
                ArrayList<Integer> context = new ArrayList<>();
                for (Integer idx : indices) {
                    context.add(line[idx]);
                }
                contexts.add(context);
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
        return new Random().nextFloat() < Math.sqrt(1e-4 / counter.get(token) * numTokens);
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
