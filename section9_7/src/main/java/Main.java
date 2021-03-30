import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslateException;

import java.io.IOException;

public class Main {

    public static NDManager manager;

    public static void main(String[] args) throws IOException, TranslateException {
        manager = NDManager.newBaseManager(Functions.tryGpu(0));



    }
}
