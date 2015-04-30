import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by mhorowitzgelb on 4/29/2015.
 */
public class CreateMonthDeathFeatures {
    public static void main(String[] args) throws IOException {
        if(args.length != 2){
            System.out.println("Please specify csv file as well as output csv file");
            return;
        }
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(args[0]));
        Instances data = loader.getDataSet();

        ArrayList<String> values = new ArrayList<String>();
        values.add("YES");
        values.add("NO");
        values.add("0");

        data.insertAttributeAt(new Attribute("DEATH12", values), 5);
        data.insertAttributeAt(new Attribute("DEATH18",values),6);
        data.insertAttributeAt(new Attribute("DEATH24",values),7);

        for(int i = 0; i < data.numInstances(); i ++){
            Instance instance = data.instance(i);
            if(instance.stringValue(4).equals("NO")){
                instance.setValue(5,"0");
                instance.setValue(6,"0");
                instance.setValue(7,"0");
            }
            else {
                if (instance.value(3) <= 365 && instance.stringValue(4).equals("YES")) {
                    instance.setValue(5, "YES");
                } else {
                    instance.setValue(5, "NO");
                }
                if (instance.value(3) <= 546 && instance.stringValue(4).equals("YES")) {
                    instance.setValue(6, "YES");
                } else {
                    instance.setValue(6, "NO");
                }
                if (instance.value(3) <= 730 && instance.stringValue(4).equals("YES")) {
                    instance.setValue(7, "YES");
                } else {
                    instance.setValue(7, "NO");
                }
            }
        }

        CSVSaver saver = new CSVSaver();
        saver.setInstances(data);
        saver.setFile(new File(args[1]));
        saver.writeBatch();
    }
}
