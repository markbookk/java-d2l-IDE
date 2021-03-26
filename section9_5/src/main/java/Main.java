import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.traces.HistogramTrace;

import java.io.*;
import java.net.URL;
import java.nio.file.*;
import java.util.*;
import java.util.stream.IntStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

public class Main {

    public static NDManager manager;

    public static void main(String[] args) throws IOException, TranslateException {
        manager = NDManager.newBaseManager(Functions.tryGpu(0));

        StringBuilder rawText = readDataNMT();
        System.out.println(rawText.substring(0, 75));

        StringBuilder text = preprocessNMT(rawText.toString());
        System.out.println(text.substring(0, 80));

        Pair<ArrayList<String[]>, ArrayList<String[]>> pair = tokenizeNMT(text.toString(), null);
        ArrayList<String[]> source = pair.getKey();
        ArrayList<String[]> target = pair.getValue();
        for (String[] subArr : source.subList(0, 6)) System.out.println(Arrays.toString(subArr));
        System.out.println("---------------");
        for (String[] subArr : target.subList(0, 6)) System.out.println(Arrays.toString(subArr));

        double[] y1 = new double[source.size()];
        for (int i = 0; i < source.size(); i++) y1[i] = source.get(i).length;
        double[] y2 = new double[target.size()];
        for (int i = 0; i < target.size(); i++) y2[i] = target.get(i).length;

        HistogramTrace trace1 =
                HistogramTrace.builder(y1).opacity(.75).name("source").nBinsX(20).build();
        HistogramTrace trace2 =
                HistogramTrace.builder(y2).opacity(.75).name("target").nBinsX(20).build();

        Layout layout = Layout.builder().barMode(Layout.BarMode.GROUP).build();
        //        Plot.show(new Figure(layout, trace1, trace2));

        Vocab srcVocab =
                new Vocab(
                        source.stream().toArray(String[][]::new),
                        2,
                        new String[] {"<pad>", "<bos>", "<eos>"});
        System.out.println(srcVocab.length());

        int[] result = truncatePad(srcVocab.getIdxs(source.get(0)), 10, srcVocab.getIdx("<pad>"));
        System.out.println(Arrays.toString(result));

        Pair<ArrayDataset, Pair<Vocab, Vocab>> output = loadDataNMT(2, 8, 600);
        ArrayDataset dataset = output.getKey();
        srcVocab = output.getValue().getKey();
        Vocab tgtVocab = output.getValue().getValue();

        for (Batch batch : dataset.getData(manager)) {
            NDArray X = batch.getData().get(0);
            NDArray xValidLen = batch.getData().get(1);
            NDArray Y = batch.getData().get(2);
            NDArray yValidLen = batch.getData().get(3);
            System.out.println(X);
            System.out.println(xValidLen);
            System.out.println(Y);
            System.out.println(yValidLen);
            break;
        }
    }

    public static StringBuilder readDataNMT() throws IOException {
        File file = new File("./fra-eng.zip");
        if (!file.exists()) {
            InputStream inputStream =
                    new URL("http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip").openStream();
            Files.copy(
                    inputStream, Paths.get("./fra-eng.zip"), StandardCopyOption.REPLACE_EXISTING);
        }

        ZipFile zipFile = new ZipFile(file);
        Enumeration<? extends ZipEntry> entries = zipFile.entries();
        InputStream stream = null;
        while (entries.hasMoreElements()) {
            ZipEntry entry = entries.nextElement();
            if (entry.getName().contains("fra.txt")) {
                stream = zipFile.getInputStream(entry);
                break;
            }
        }

        String[] lines;
        try (BufferedReader in = new BufferedReader(new InputStreamReader(stream))) {
            lines = in.lines().toArray(String[]::new);
        }
        StringBuilder output = new StringBuilder();
        for (int i = 0; i < lines.length; i++) {
            output.append(lines[i] + "\n");
        }
        return output;
    }

    public static StringBuilder preprocessNMT(String text) {
        // Replace non-breaking space with space, and convert uppercase letters to
        // lowercase ones

        text = text.replace('\u202f', ' ').replaceAll("\\xa0", " ").toLowerCase();

        // Insert space between words and punctuation marks
        StringBuilder out = new StringBuilder();
        Character currChar;
        for (int i = 0; i < text.length(); i++) {
            currChar = text.charAt(i);
            if (i > 0 && noSpace(currChar, text.charAt(i - 1))) {
                out.append(' ');
            }
            out.append(currChar);
        }
        return out;
    }

    public static boolean noSpace(Character currChar, Character prevChar) {
        /* Preprocess the English-French dataset. */
        return new HashSet<>(Arrays.asList(',', '.', '!', '?')).contains(currChar)
                && prevChar != ' ';
    }

    public static Pair<ArrayList<String[]>, ArrayList<String[]>> tokenizeNMT(
            String text, Integer numExamples) {
        ArrayList<String[]> source = new ArrayList<>();
        ArrayList<String[]> target = new ArrayList<>();

        int i = 0;
        for (String line : text.split("\n")) {
            if (numExamples != null && i > numExamples) {
                break;
            }
            String[] parts = line.split("\t");
            if (parts.length == 2) {
                source.add(parts[0].split(" "));
                target.add(parts[1].split(" "));
            }
            i += 1;
        }
        return new Pair<>(source, target);
    }

    public static int[] truncatePad(Integer[] integerLine, int numSteps, int paddingToken) {
        /* Truncate or pad sequences */
        int[] line = Arrays.stream(integerLine).mapToInt(i -> i).toArray();
        if (line.length > numSteps) return Arrays.copyOfRange(line, 0, numSteps);
        int[] paddingTokenArr = new int[numSteps - line.length]; // Pad
        for (int i = 0; i < paddingTokenArr.length; i++) paddingTokenArr[i] = paddingToken;

        return IntStream.concat(Arrays.stream(line), Arrays.stream(paddingTokenArr)).toArray();
    }

    public static Pair<NDArray, NDArray> buildArrayNMT(
            ArrayList<String[]> lines, Vocab vocab, int numSteps) {
        /* Transform text sequences of machine translation into minibatches. */
        ArrayList<Integer[]> linesIntArr = new ArrayList<>();
        for (int i = 0; i < lines.size(); i++) linesIntArr.add(vocab.getIdxs(lines.get(i)));
        for (int i = 0; i < linesIntArr.size(); i++) {
            ArrayList<Integer> temp = new ArrayList<>();
            temp.addAll(Arrays.asList(linesIntArr.get(i)));
            temp.add(vocab.getIdx("<eos>"));
            linesIntArr.set(i, temp.stream().toArray(n -> new Integer[n]));
        }

        NDArray arr = manager.create(new Shape(linesIntArr.size(), numSteps), DataType.INT32);
        int row = 0;
        for (Integer[] line : linesIntArr) {
            NDArray rowArr = manager.create(truncatePad(line, numSteps, vocab.getIdx("<pad>")));
            arr.set(new NDIndex("{}:", row), rowArr);
            row += 1;
        }
        NDArray validLen = arr.neq(vocab.getIdx("<pad>")).sum(new int[] {1});
        return new Pair<>(arr, validLen);
    }

    public static Pair<ArrayDataset, Pair<Vocab, Vocab>> loadDataNMT(
            int batchSize, int numSteps, int numExamples) throws IOException {
        /* Return the iterator and the vocabularies of the translation dataset. */
        StringBuilder text = preprocessNMT(readDataNMT().toString());
        Pair<ArrayList<String[]>, ArrayList<String[]>> pair =
                tokenizeNMT(text.toString(), numExamples);
        ArrayList<String[]> source = pair.getKey();
        ArrayList<String[]> target = pair.getValue();
        Vocab srcVocab =
                new Vocab(
                        source.stream().toArray(String[][]::new),
                        2,
                        new String[] {"<pad>", "<bos>", "<eos>"});
        Vocab tgtVocab =
                new Vocab(
                        target.stream().toArray(String[][]::new),
                        2,
                        new String[] {"<pad>", "<bos>", "<eos>"});

        Pair<NDArray, NDArray> pairArr = buildArrayNMT(source, srcVocab, numSteps);
        NDArray srcArr = pairArr.getKey();
        NDArray srcValidLen = pairArr.getValue();

        pairArr = buildArrayNMT(target, tgtVocab, numSteps);
        NDArray tgtArr = pairArr.getKey();
        NDArray tgtValidLen = pairArr.getValue();

        ArrayDataset dataset =
                new ArrayDataset.Builder()
                        .setData(srcArr, srcValidLen, tgtArr, tgtValidLen)
                        .setSampling(batchSize, true)
                        .build();

        return new Pair<>(dataset, new Pair<>(srcVocab, tgtVocab));
    }
}
