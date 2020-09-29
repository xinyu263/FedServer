package com.fed.train.model;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class TaskTest {


    public static int seed=123456;
    public static int inX=100;
    public static int inY=100;
    public static int outputNum=5;

    public static void main(String[] args) {

        MultiLayerConfiguration config=new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Nesterovs(0.1, 0.9))
                .list()
                .layer(0,new DenseLayer.Builder()
                    .nIn(inX*inY)
                    .nOut(500)
                    .build())
                .layer(new DenseLayer.Builder() //create the second input layer
                        .nIn(500)
                        .nOut(100)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                        .activation(Activation.SOFTMAX)
                        .nOut(outputNum)
                        .build())
                .build();





    }


}
