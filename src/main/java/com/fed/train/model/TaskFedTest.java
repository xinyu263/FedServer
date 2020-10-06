package com.fed.train.model;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class TaskFedTest {
    private static Logger log = LoggerFactory.getLogger(TaskFedTest.class);
    public static void main(String[] args) throws IOException {
        int height = 370;  // 输入图像高度
        int width = 1224;   // 输入图像宽度
        int channels = 1; // 输入图像通道数
        int outputNum = 5; //分类
        int batchSize = 64;//64
        int nEpochs = 10;//本地迭代次数
        int seed = 1234;
        Random randNumGen = new Random(seed);

        String inputDataDir="D:\\Data\\dataOnAndroid\\";
        int userNum=1;//10，参与方数量
        int globalIteration=1;

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        //训练数据的向量化
        ArrayList<File> trainData=new ArrayList<>();
        ArrayList<FileSplit> trainSplit=new ArrayList<>();
        ArrayList<ImageRecordReader> trainRR=new ArrayList<>();
        ArrayList<DataSetIterator> trainIter=new ArrayList<>();

        for(int i=0;i<userNum;i++){
            trainData.add(new File(inputDataDir + "user"+i));
            trainSplit.add(new FileSplit(trainData.get(i), NativeImageLoader.ALLOWED_FORMATS, randNumGen));
            trainRR.add(new ImageRecordReader(height, width, channels, labelMaker));
            trainRR.get(i).initialize(trainSplit.get(i));
            trainIter.add(new RecordReaderDataSetIterator(trainRR.get(i), batchSize, 1, outputNum));
            scaler.fit(trainIter.get(i));
            trainIter.get(i).setPreProcessor(scaler);
        }


        //测试数据的向量化
        File testData = new File(inputDataDir + "test");
        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
        testRR.initialize(testSplit);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
        testIter.setPreProcessor(scaler);


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0005)
                .updater(new Adam(0.0001))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .nIn(channels)
                        .stride(2, 2)
                        .nOut(12)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new BatchNormalization.Builder()
                        .nIn(12)
                        .nOut(12)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .nIn(12)
                        .stride(2, 2)
                        .nOut(24)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(3, new BatchNormalization.Builder()
                        .nIn(24)
                        .nOut(24)
                        .build())
                .layer(4, new ConvolutionLayer.Builder(3, 3)
                        .nIn(24)
                        .stride(2, 2)
                        .nOut(48)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(5, new BatchNormalization.Builder()
                        .nIn(48)
                        .nOut(48)
                        .build())
                .layer(6, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(128).build())
                .layer(7, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(height, width, 3)) // InputType.convolutional for normal image
                .build();



        log.info("create model....");
        //创建各参与者模型
        ArrayList<MultiLayerNetwork> nets=new ArrayList<>();
        for (int i=0;i<userNum;i++){
            nets.add(new MultiLayerNetwork(conf));
            nets.get(i).init();
            nets.get(i).setListeners(new ScoreIterationListener(10), new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END));
        }

        log.info("train model....");
        //训练
        for(int i=0;i<globalIteration;i++){
            for(int j=0;j<userNum;j++){
                log.info("train model "+j);
                    nets.get(j).fit(trainIter.get(j),nEpochs);
            }
        }


/*        //测试
        log.info("Evaluate model....");
        ArrayList<Evaluation> evals=new ArrayList<>();
        for(int i=0;i<userNum;i++){
            log.info("Evaluate model "+i);
            evals.add(nets.get(i).evaluate(testIter));
            //log.info(evals.get(i).stats());
        }*/
        
    }
}
