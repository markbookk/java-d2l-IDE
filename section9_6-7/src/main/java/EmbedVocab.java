import ai.djl.modality.nlp.Vocabulary;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class EmbedVocab implements Vocabulary {

    private Map<Long, String> tokenMap = new ConcurrentHashMap<>();
    private Map<String, Long> indicesMap = new ConcurrentHashMap<>();
    private long index;

    /** {@inheritDoc} */
    @Override
    public String getToken(long index) {
        return tokenMap.get(index);
    }

    /** {@inheritDoc} */
    @Override
    public long getIndex(String token) {
        if (!indicesMap.containsKey(token)) {
            indicesMap.put(token, index);
            tokenMap.put(index++, token);
        }
        return indicesMap.get(token);
    }

    /** {@inheritDoc} */
    @Override
    public boolean contains(String token) {
        return indicesMap.containsKey(token);
    }

    /** {@inheritDoc} */
    @Override
    public long size() {
        return indicesMap.size();
    }
}