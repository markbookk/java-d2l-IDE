import org.apache.commons.math3.distribution.EnumeratedDistribution;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.util.Pair;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class RandomGenerator {
    /* Draw a random int in [0, n] according to n sampling weights. */

    private List<Integer> population;
    private List<Double> samplingWeights;
    private List<Integer> candidates;
    private List<Pair<Integer, Double>> pmf;
    private int i;

    public RandomGenerator(List<Double> samplingWeights) {
        this.population =
                IntStream.range(0, samplingWeights.size()).boxed().collect(Collectors.toList());
        this.samplingWeights = samplingWeights;
        this.candidates = new ArrayList<>();
        this.i = 0;

        this.pmf = new ArrayList<>();
        for (int i = 0; i < samplingWeights.size(); i++) {
            this.pmf.add(new Pair(this.population.get(i), this.samplingWeights.get(i).doubleValue()));
        }
    }

    public Integer draw() {
        if (this.i == this.candidates.size()) {
            this.candidates =
                    Arrays.asList((Integer[]) new EnumeratedDistribution(this.pmf).sample(10000, new Integer[] {}));
            this.i = 0;
        }
        this.i += 1;
        return this.candidates.get(this.i - 1);
    }
}
