import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.TranslateException;
import ai.djl.util.Progress;

import java.io.IOException;

public class TimeMachineDataset extends RandomAccessDataset {

    private Vocab vocab;

    public TimeMachineDataset(Builder builder) {
        super(builder);
    }

    @Override
    public Record get(NDManager manager, long index) throws IOException {
        return null;
    }

    @Override
    protected long availableSize() {
        return 0;
    }

    @Override
    public void prepare(Progress progress) throws IOException, TranslateException {

    }

    public Vocab getVocab() {
        return this.vocab;
    }

    public static final class Builder extends BaseBuilder<Builder> {

        int numSteps;
        int maxTokens;


        @Override
        protected Builder self() { return this; }

        public Builder setSteps(int steps) {
            this.numSteps = steps;
            return this;
        }

        public Builder setMaxTokens(int maxTokens) {
            this.maxTokens = maxTokens;
            return this;
        }

        public TimeMachineDataset build() {
            return new TimeMachineDataset(this);
        }
    }
}
