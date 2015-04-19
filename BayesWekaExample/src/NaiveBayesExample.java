import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
This is an example of how to run Naive bayes
 It won't work unless you use the lymph training from HW2
 */
public class NaiveBayesExample {
    public static void main(String[] args) throws Exception {
        if(args.length != 2){
            System.out.println("Please specify training arff file and testing arff file");
            return;
        }

        //Import ARFF files
        DataSource trainingSource = new DataSource(args[0]);
        DataSource testingSource = new DataSource(args[1]);


        //Load them into instances
        Instances trainingInstances = trainingSource.getDataSet();
        Instances testingInstances = testingSource.getDataSet();

        //Set the class index for the instances in our case it should be metastases
        trainingInstances.setClassIndex(18);
        testingInstances.setClassIndex(18);

        //Make the classifier and build it from training set
        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(trainingInstances);

        //Make an evaluator and evaluate the testing set
        Evaluation evaluation = new Evaluation(trainingInstances);
        evaluation.evaluateModel(naiveBayes, testingInstances);


        //Print out a summary of the results
        //Check the evaluation class for other things you might want to test.
        System.out.println(evaluation.toSummaryString());

    }
}
