import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * Created by maxhorowitz on 4/19/15.
 */
public class SMOExample {
    public static void main(String[] args) throws Exception {
        if(args.length != 2){
            System.out.println("Please input lymph_train.arff and lymph_test.arff file locations");
            return;
        }
        Instances trainingSet = new Instances(new BufferedReader(new FileReader(args[0])));
        Instances testSet = new Instances(new BufferedReader(new FileReader(args[1])));
        trainingSet.setClassIndex(trainingSet.numAttributes() -1);
        testSet.setClassIndex(testSet.numAttributes() -1);


        SMO model = new SMO();

        model.buildClassifier(trainingSet);

        Evaluation eval = new Evaluation(trainingSet);

        eval.evaluateModel(model, testSet);

        System.out.println(eval.toSummaryString());

    }
}
