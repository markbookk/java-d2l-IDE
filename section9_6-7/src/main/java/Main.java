import ai.djl.ndarray.*;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.translate.TranslateException;

import java.io.IOException;

public class Main {

    public static NDManager manager;

    public static void main(String[] args) throws IOException, TranslateException {
        manager = NDManager.newBaseManager(Functions.tryGpu(0));

        Seq2SeqEncoder encoder = new Seq2SeqEncoder(10, 8, 16, 2, 0);
        NDArray X = manager.zeros(new Shape(4, 7));
        NDList outputState =
                encoder.forward(new ParameterStore(manager, false), new NDList(X), false);

        NDArray output = outputState.get(0);
        System.out.println(output.getShape());

        NDList state = outputState.subNDList(1);
        System.out.println(state.size());
        System.out.println(state.get(0).getShape());

        Seq2SeqDecoder decoder = new Seq2SeqDecoder(10, 8, 16, 2, 0);
        state = decoder.beginState(outputState);
        outputState =
                decoder.forward(
                        new ParameterStore(manager, false), new NDList(X).addAll(state), false);

        output = outputState.get(0);
        System.out.println(output.getShape());

        state = outputState.subNDList(1);
        System.out.println(state.size());
        System.out.println(state.get(0).getShape());

        X = manager.create(new int[][] {{1, 2, 3}, {4, 5, 6}});
        System.out.println(X.sequenceMask(manager.create(new int[] {1, 2})));

    }
}
