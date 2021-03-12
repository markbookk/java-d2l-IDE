/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.recurrent.RNN;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public class RNNModel extends AbstractBlock {

    private static final byte VERSION = 2;
    private RNN rnnLayer;
    private Linear dense;
    private int vocabSize;

    public RNNModel(RNN rnnLayer, int vocabSize) {
        super(VERSION);
        this.rnnLayer = rnnLayer;
        this.addChildBlock("rnn", rnnLayer);
        this.vocabSize = vocabSize;
        this.dense = Linear.builder().setUnits(vocabSize).build();
        this.addChildBlock("linear", dense);
    }

    /** {@inheritDoc} */
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        /* rnnLayer is already initialized so we don't have to do anything here, just override it.*/
//        rnnLayer.clear();
//        rnnLayer.setInitializer(Initializer.ZEROS, Parameter.Type.WEIGHT);
//        rnnLayer.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
//
//        dense.clear();
//        dense.setInitializer(Initializer.ZEROS, Parameter.Type.WEIGHT);
//        dense.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
    }

//        List<Block> childs = getChildren().values();
//        for (Block child : childs) {
//            for (Parameter parameter : child.getParameters().values()) {
//                parameter.setInitializer(new UniformInitializer());
//            }
//        }
//        rnnLayer.clear();
//        rnnLayer.initialize(manager, dataType, inputShapes);
//
//        inputShapes = rnnLayer.getOutputShapes(inputShapes);
//        int shapeLength = inputShapes[0].getShape().length;
//        Shape linearShape = new Shape(-1, inputShapes[0].get(shapeLength-1));
//        dense.clear();
//        dense.initialize(manager, dataType, linearShape);
//    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray X = inputs.get(0).transpose().oneHot(this.vocabSize);
        inputs.set(0, X);
        NDList result = this.rnnLayer.forward(parameterStore, inputs, training);
        NDArray Y = result.get(0);
        NDArray state = result.get(1);

        int shapeLength = Y.getShape().getShape().length;
        NDList output = this.dense.forward(parameterStore, new NDList(Y
                .reshape(new Shape(-1, Y.getShape().get(shapeLength-1)))), training);
        return new NDList(output.get(0), state);
    }


    /* We won't implement this since we won't be using it but it's required as part of an AbstractBlock  */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[0];
    }
}
