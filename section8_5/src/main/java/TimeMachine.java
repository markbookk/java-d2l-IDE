import ai.djl.util.Pair;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

public class TimeMachine {
    /**
     * Split text lines into word or character tokens.
     */
    public static String[][] tokenize(String[] lines, String token) throws Exception {
        String[][] output = new String[lines.length][];
        if (token == "word") {
            for (int i = 0; i < output.length; i++) {
                output[i] = lines[i].split(" ");
            }
        }else if (token == "char") {
            for (int i = 0; i < output.length; i++) {
                output[i] = lines[i].split("");
            }
        }else {
            throw new Exception("ERROR: unknown token type: " + token);
        }
        return output;
    }

    /**
     * Read `The Time Machine` dataset and return an array of the lines
     */
    public static String[] readTimeMachine() throws IOException {
        URL url = new URL("http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt");
        String[] lines;
        try (BufferedReader in = new BufferedReader(new InputStreamReader(url.openStream()))) {
            lines = in.lines().toArray(String[]::new);
        }

        for (int i = 0; i < lines.length; i++) {
            lines[i] = lines[i].replaceAll("[^A-Za-z]+", " ").strip().toLowerCase();
        }
        return lines;
    }

    /**
     * Return token indices and the vocabulary of the time machine dataset.
     */
    public static Pair<List<Integer>, Vocab> loadCorpusTimeMachine(int maxTokens) throws IOException, Exception {
        String[] lines = readTimeMachine();
        String[][] tokens = tokenize(lines, "char");
        Vocab vocab = new Vocab(tokens, 0, new String[0]);
        // Since each text line in the time machine dataset is not necessarily a
        // sentence or a paragraph, flatten all the text lines into a single list
        List<Integer> corpus = new ArrayList<>();
        for (int i = 0; i < tokens.length; i++) {
            for (int j = 0; j < tokens[i].length; j++) {
                if (tokens[i][j] != "") {
                    corpus.add(vocab.getIdx(tokens[i][j]));
                }
            }
        }
        if (maxTokens > 0) {
            corpus = corpus.subList(0, maxTokens);
        }
        return new Pair(corpus, vocab);
    }
}