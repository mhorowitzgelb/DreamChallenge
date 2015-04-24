import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

/**
 * Created by mhorowitzgelb on 4/19/2015.
 */
public class NeuralNetworkExample {
    public static void main(String[] args) throws Exception {
        if(args.length != 1){
            return;
        }
        Instances trainingInstances = new Instances(new BufferedReader(new FileReader(args[0])));
        trainingInstances.setClassIndex(trainingInstances.numAttributes() -1);

        MultilayerPerceptron model = new MultilayerPerceptron();

        //Specify the node size of each hidden layer
        model.setHiddenLayers("5");
        model.setNormalizeAttributes(true);
        model.buildClassifier(trainingInstances);

        Evaluation eval = new Evaluation(trainingInstances);
        eval.crossValidateModel(model,trainingInstances,10,new Random(1));

        System.out.println(eval.toSummaryString());



    }
}
