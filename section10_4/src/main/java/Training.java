import ai.djl.metric.Metrics;
import ai.djl.ndarray.*;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;

import java.io.IOException;
import java.util.Map;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

class Training {

    public static NDArray linreg(NDArray X, NDArray w, NDArray b) {
        return X.dot(w).add(b);
    }

    public static NDArray squaredLoss(NDArray yHat, NDArray y) {
        return (yHat.sub(y.reshape(yHat.getShape()))).mul
                ((yHat.sub(y.reshape(yHat.getShape())))).div(2);
    }

    public static void sgd(NDList params, float lr, int batchSize) {
        for (int i = 0; i < params.size(); i++) {
            NDArray param = params.get(i);
            // Update param in place.
            // param = param - param.gradient * lr / batchSize
            param.subi(param.getGradient().mul(lr).div(batchSize));
        }
    }

    /** Allow to do gradient calculations on subManager **/
    public static void sgd(NDList params, float lr, int batchSize, NDManager subManager) {
        for (int i = 0; i < params.size(); i++) {
            NDArray param = params.get(i);
            // Update param in place.
            // param = param - param.gradient * lr / batchSize
            NDArray gradient = param.getGradient();
            gradient.attach(subManager);
            param.subi(gradient.mul(lr).div(batchSize));
        }
    }

    /** Clip the gradient. */
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
        if (norm > theta) {
            for (NDArray param : params) {
                NDArray gradient = param.getGradient();
                gradient.muli(theta / norm);
            }
        }
    }

    public static float accuracy(NDArray yHat, NDArray y) {
        // Check size of 1st dimension greater than 1
        // to see if we have multiple samples
        if (yHat.getShape().size(1) > 1) {
            // Argmax gets index of maximum args for given axis 1
            // Convert yHat to same dataType as y (int32)
            // Sum up number of true entries
            return yHat.argMax(1).toType(DataType.INT32, false).eq(y.toType(DataType.INT32, false))
                    .sum().toType(DataType.FLOAT32, false).getFloat();
        }
        return yHat.toType(DataType.INT32, false).eq(y.toType(DataType.INT32, false))
                .sum().toType(DataType.FLOAT32, false).getFloat();
    }

    public static void trainingChapter6(ArrayDataset trainIter, ArrayDataset testIter,
                                        int numEpochs, Trainer trainer, Map<String, double[]> evaluatorMetrics, double avgTrainTimePerEpoch) throws IOException, TranslateException {

        trainer.setMetrics(new Metrics());

        EasyTrain.fit(trainer, numEpochs, trainIter, testIter);

        Metrics metrics = trainer.getMetrics();

        trainer.getEvaluators().stream()
                .forEach(evaluator -> {
                    evaluatorMetrics.put("train_epoch_" + evaluator.getName(), metrics.getMetric("train_epoch_" + evaluator.getName()).stream()
                            .mapToDouble(x -> x.getValue().doubleValue()).toArray());
                    evaluatorMetrics.put("validate_epoch_" + evaluator.getName(), metrics.getMetric("validate_epoch_" + evaluator.getName()).stream()
                            .mapToDouble(x -> x.getValue().doubleValue()).toArray());
                });

        avgTrainTimePerEpoch = metrics.mean("epoch");
    }

    /* Softmax-regression-scratch */
    public static float evaluateAccuracy(UnaryOperator<NDArray> net, Iterable<Batch> dataIterator) {
        Accumulator metric = new Accumulator(2);  // numCorrectedExamples, numExamples
        for (Batch batch : dataIterator) {
            NDArray X = batch.getData().head();
            NDArray y = batch.getLabels().head();
            metric.add(new float[]{accuracy(net.apply(X), y), (float)y.size()});
            batch.close();
        }
        return metric.get(0) / metric.get(1);
    }
    /* End Softmax-regression-scratch */

    /* MLP */
    /* Evaluate the loss of a model on the given dataset */
    public static float evaluateLoss(UnaryOperator<NDArray> net, Iterable<Batch> dataIterator, BinaryOperator<NDArray> loss) {
        Accumulator metric = new Accumulator(2);  // sumLoss, numExamples

        for (Batch batch : dataIterator) {
            NDArray X = batch.getData().head();
            NDArray y = batch.getLabels().head();
            metric.add(new float[]{loss.apply(net.apply(X), y).sum().getFloat(), (float) y.size()});
            batch.close();
        }
        return metric.get(0) / metric.get(1);
    }
    /* End MLP */
}
