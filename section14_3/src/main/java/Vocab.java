import org.checkerframework.checker.units.qual.A;

import java.util.*;

public class Vocab {
    public int unk;
    public List<Map.Entry<String, Integer>> tokenFreqs;
    public List<String> idxToToken;
    public HashMap<String, Integer> tokenToIdx;

    public Vocab(String[][] tokens, int minFreq, String[] reservedTokens) {
        // Sort according to frequencies
        LinkedHashMap<Object, Integer> counterObject = countCorpus2D(tokens);
        LinkedHashMap<String, Integer> counter = new LinkedHashMap<>();
        for (Map.Entry<Object, Integer> e : counterObject.entrySet()) {
            counter.put((String) e.getKey(), e.getValue());
        }

        this.tokenFreqs = new ArrayList<Map.Entry<String, Integer>>(counter.entrySet());
        Collections.sort(
                tokenFreqs,
                new Comparator<Map.Entry<String, Integer>>() {
                    public int compare(
                            Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
                        return (o2.getValue()).compareTo(o1.getValue());
                    }
                });

        // The index for the unknown token is 0
        this.unk = 0;
        List<String> uniqTokens = new ArrayList<>();
        uniqTokens.add("<unk>");
        Collections.addAll(uniqTokens, reservedTokens);
        for (Map.Entry<String, Integer> entry : tokenFreqs) {
            if (entry.getValue() >= minFreq && !uniqTokens.contains(entry.getKey())) {
                uniqTokens.add(entry.getKey());
            }
        }

        this.idxToToken = new ArrayList<>();
        this.tokenToIdx = new HashMap<>();
        for (String token : uniqTokens) {
            this.idxToToken.add(token);
            this.tokenToIdx.put(token, this.idxToToken.size() - 1);
        }
    }

    public int length() {
        return this.idxToToken.size();
    }

    public Integer[] getIdxs(String[] tokens) {
        List<Integer> idxs = new ArrayList<>();
        for (String token : tokens) {
            idxs.add(getIdx(token));
        }
        return idxs.toArray(new Integer[0]);
    }

    public Integer getIdx(String token) {
        return this.tokenToIdx.getOrDefault(token, this.unk);
    }

    public List<String> toTokens(List<Integer> indices) {
        List<String> tokens = new ArrayList<>();
        for (Integer index : indices) {
            tokens.add(toToken(index));
        }
        return tokens;
    }

    public String toToken(Integer index) {
        return this.idxToToken.get(index);
    }

    /** Count token frequencies. */
    public static LinkedHashMap<Object, Integer> countCorpus(Object[] tokens) {

        LinkedHashMap<Object, Integer> counter = new LinkedHashMap<>();
        if (tokens.length != 0) {
            for (Object token : tokens) {
                counter.put(token, counter.getOrDefault(token, 0) + 1);
            }
        }
        return counter;
    }

    /** Flatten a list of token lists into a list of tokens */
    public static LinkedHashMap<Object, Integer> countCorpus2D(Object[][] tokens) {
        List<Object> allTokens = new ArrayList<Object>();
        for (int i = 0; i < tokens.length; i++) {
            for (int j = 0; j < tokens[i].length; j++) {
                if (tokens[i][j] != "") {
                    allTokens.add(tokens[i][j]);
                }
            }
        }
        return countCorpus(allTokens.toArray(new Object[0]));
    }
}
