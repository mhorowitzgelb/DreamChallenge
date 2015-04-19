import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.search.global.TAN;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 * Classifier using TAN
 */
public class TANExample {

    public static void main(String[] args) throws Exception {
        if(args.length != 2){
            System.out.println("please specify a lymph_train.arff and lymph_test.arff locations");
            return;
        }
        Instances trainingInstances = new Instances(new BufferedReader(new FileReader(args[0])));
        Instances testingInstances = new Instances(new BufferedReader(new FileReader(args[1])));
        trainingInstances.setClassIndex(trainingInstances.numAttributes() -1);
        testingInstances.setClassIndex(testingInstances.numAttributes() -1);

        BayesNet bayesNet = new BayesNet();

        //Object for finding maximum weight spanning tree using Chow Liu
        //You can set options for how weights are scored also there's something
        //weird you can do with Markhov Blankets that you can do to correct structure
        //I am just leaving it default

        //Also this is global search TAN which is slightly different than local search TAN
        TAN tan = new TAN();
        bayesNet.setSearchAlgorithm(tan);
        bayesNet.buildClassifier(trainingInstances);

        Evaluation eval = new Evaluation(trainingInstances);
        eval.evaluateModel(bayesNet,testingInstances);

        System.out.println(eval.toSummaryString());

    }
}
