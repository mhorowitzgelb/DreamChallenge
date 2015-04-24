import com.sun.deploy.util.ArrayUtil;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.pmml.jaxbbindings.DecisionTree;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

/**
 * Created by mhorowitzgelb on 4/19/2015.
 */
public class TestCSVToARFFConversion {
    public static void main(String[] args) throws Exception {
        if(args.length != 2){
            System.out.println("Give input csv and output arff files");
        }

        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(args[0]));
        Instances data = loader.getDataSet();
        data.setClassIndex(4);
        /*
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(args[1]));
        saver.writeBatch();*/


        Remove remove = new Remove();
        ArrayList<Integer> removeList = new ArrayList<Integer>();
        for(int i =0; i < data.numAttributes(); i ++){
            Attribute attribute = data.attribute(i);
            if(attribute.type() == Attribute.STRING)
                removeList.add(i);
        }
        int[] array = new int[removeList.size()];
        for(int i = 0; i < removeList.size(); i ++){
            array[i] = removeList.get(i);
        }
        remove.setAttributeIndicesArray(array);
        remove.setInputFormat(data);
        Instances filteredInstances = Filter.useFilter(data,remove);
        J48 tree = new J48();
        tree.buildClassifier(filteredInstances);




        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(tree, data, 10, new Random(1));

        System.out.println(eval.toSummaryString());


    }
}
