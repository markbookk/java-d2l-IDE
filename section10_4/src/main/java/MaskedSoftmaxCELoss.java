import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;

public class MaskedSoftmaxCELoss extends SoftmaxCrossEntropyLoss {
    /* The softmax cross-entropy loss with masks. */

    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        NDArray weights = labels.head().onesLike().expandDims(-1).sequenceMask(labels.get(1));
        // Remove the weights from the labels NDList because otherwise, it will throw an error as SoftmaxCrossEntropyLoss
        // expects only one NDArray for label and one NDArray for prediction
        labels.remove(1);
        return super.evaluate(labels, predictions).mul(weights).mean(new int[] {1});
    }
}
