import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.IOException;
import java.util.Scanner;

/**
 * Created by mhorowitzgelb on 5/5/2015.
 */
public class FuckingBayesFix {
    public static void main(String[] args) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("C:\\Users\\mhorowitzgelb\\Documents\\DreamChallenge\\LeaderBoard\\ouput.csv"));
        Instances instances = loader.getDataSet();
        for(Instance i : instances){
            if(i.stringValue(0).equals("1:YES")){
                System.out.println(i.value(1));
            }
            else if(i.stringValue(0).equals("2:NO")){
                System.out.println(1 - i.value(1));
            }
            else{
                throw new Exception(i.stringValue(0));
            }
        }
    }
}
