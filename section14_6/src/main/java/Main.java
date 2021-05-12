import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.*;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.*;
import ai.djl.nn.core.Embedding;
import ai.djl.training.*;
import ai.djl.training.dataset.*;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import ai.djl.util.ZipUtils;

import java.io.*;
import java.net.URL;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.*;

public class Main {

    public static NDManager manager;

    public static void main(String[] args) throws IOException, TranslateException {
        manager = NDManager.newBaseManager(Functions.tryGpu(0));

        String[] symbols =
                new String[] {
                    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
                    "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "_", "[UNK]"
                };
        HashMap<String, Integer> rawTokenFreqs = new HashMap<>();
        rawTokenFreqs.put("fast_", 4);
        rawTokenFreqs.put("faster_", 3);
        rawTokenFreqs.put("tall_", 5);
        rawTokenFreqs.put("taller_", 4);

        LinkedHashMap<String, Integer> tokenFreqs = new LinkedHashMap<>();
        for (Map.Entry<String, Integer> e : rawTokenFreqs.entrySet()) {
            String token = e.getKey();
            tokenFreqs.put(String.join(" ", token.split("")), rawTokenFreqs.get(token));
        }
        System.out.println(tokenFreqs);

        int numMerges = 10;
        for (int i = 0; i < numMerges; i++) {
            Pair<String, String> maxFreqPair = getMaxFreqPair(tokenFreqs);
            Pair<LinkedHashMap<String, Integer>, String[]> pair =
                    mergeSymbols(maxFreqPair, tokenFreqs);
            tokenFreqs = pair.getKey();
            symbols =
                    Stream.concat(Arrays.stream(symbols), Arrays.stream(pair.getValue()))
                            .toArray(String[]::new);
            System.out.println(
                    "Merge #"
                            + (i + 1)
                            + ": ("
                            + maxFreqPair.getKey()
                            + ", "
                            + maxFreqPair.getValue()
                            + ")");
        }

        System.out.println(Arrays.toString(symbols));

        System.out.print(tokenFreqs.keySet());

        String[] tokens = new String[] {"tallest_", "fatter_"};
        System.out.println(segmentBPE(tokens, symbols));
    }

    public static ArrayList<String> segmentBPE(String[] tokens, String[] symbols) {
        ArrayList<String> outputs = new ArrayList<>();
        for (String token : tokens) {
            int start = 0;
            int end = token.length();
            ArrayList<String> curOutput = new ArrayList<>();
            // Segment token with the longest possible subwords from symbols
            while (start < token.length() && start < end) {
                if (Arrays.asList(symbols).contains(token.substring(start, end))) {
                    curOutput.add(token.substring(start, end));
                    start = end;
                    end = token.length();
                } else {
                    end -= 1;
                }
            }
            if (start < tokens.length) {
                curOutput.add("[UNK]");
            }
            String temp = "";
            for (String s : curOutput) temp += s;
            outputs.add(temp);
        }
        return outputs;
    }

    public static Pair<String, String> getMaxFreqPair(LinkedHashMap<String, Integer> tokenFreqs) {
        LinkedHashMap<Pair<String, String>, Integer> pairs = new LinkedHashMap<>();
        for (Map.Entry<String, Integer> e : tokenFreqs.entrySet()) {
            // Key of 'pairs' is a tuple of two consecutive symbols
            String token = e.getKey();
            Integer freq = e.getValue();
            String[] symbols = token.split(" ");
            for (int i = 0; i < symbols.length - 1; i++) {
                pairs.put(
                        new Pair<>(symbols[i], symbols[i + 1]),
                        pairs.getOrDefault(new Pair<>(symbols[i], symbols[i + 1]), 0) + freq);
            }
        }
        int max = 0; // Key of `pairs` with the max value
        Pair<String, String> maxFreqPair = null;
        for (Map.Entry<Pair<String, String>, Integer> pair : pairs.entrySet()) {
            if (max < pair.getValue()) {
                max = pair.getValue();
                maxFreqPair = pair.getKey();
            }
        }
        return maxFreqPair;
    }

    public static Pair<LinkedHashMap<String, Integer>, String[]> mergeSymbols(
            Pair<String, String> maxFreqPair, LinkedHashMap<String, Integer> tokenFreqs) {
        ArrayList<String> symbols = new ArrayList<>();
        symbols.add(maxFreqPair.getKey() + maxFreqPair.getValue());

        LinkedHashMap<String, Integer> newTokenFreqs = new LinkedHashMap<>();
        for (Map.Entry<String, Integer> e : tokenFreqs.entrySet()) {
            String token = e.getKey();
            String newToken =
                    token.replace(
                            maxFreqPair.getKey() + " " + maxFreqPair.getValue(),
                            maxFreqPair.getKey() + "" + maxFreqPair.getValue());
            newTokenFreqs.put(newToken, tokenFreqs.get(token));
        }
        return new Pair(newTokenFreqs, symbols.toArray(new String[symbols.size()]));
    }
}
