import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Instances;
import weka.core.Utils;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

/**
 * This example demonstrates how to do ROC curves with cross validation and graph the results
 * Use the lymp_train.arff
 */
public class ROCCurve {
    public static void main(String[] args) throws Exception {
        if(args.length != 1){
            System.out.println("Please specify the lymp_train.arff file location");
        }

        Instances data = new Instances( new BufferedReader(new FileReader(args[0])));
        data.setClassIndex(data.numAttributes() -1);

        Classifier classifier = new NaiveBayes();
        Evaluation eval = new Evaluation(data);

        //Cross validate with 10 folds
        eval.crossValidateModel(classifier, data, 10, new Random(1));

        //generate curve
        ThresholdCurve thresholdCurve = new ThresholdCurve();
        Instances result = thresholdCurve.getCurve(eval.predictions());

        //Plot curve
        ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
        vmc.setROCString("(Area under ROC = " + Utils.doubleToString(thresholdCurve.getROCArea(result),4) + ")");
        vmc.setName(result.relationName());
        PlotData2D tempd = new PlotData2D(result);
        tempd.setPlotName(result.relationName());
        tempd.addInstanceNumberAttribute();

        //Specify which points in the graph should be connected
        boolean[] cp = new boolean[result.numInstances()];
        for(int i = 0 ; i < cp.length; i ++){
            cp[i] = true;
        }
        tempd.setConnectPoints(cp);
        vmc.addPlot(tempd);

        //Display curve
        String plotName = vmc.getName();

        final JFrame jf = new JFrame("Weka Classifier visualize:" + plotName);
        jf.setSize(500,400);
        jf.getContentPane().setLayout(new BorderLayout());
        jf.getContentPane().add(vmc, BorderLayout.CENTER);
        jf.addWindowListener(new WindowAdapter() {


            @Override
            public void windowClosing(WindowEvent e){
                jf.dispose();
                System.exit(0);
            }
        });

        jf.setVisible(true);

    }
}
