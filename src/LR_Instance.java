import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Iterator;

/**
 * Created by chzhenzh on 5/16/2016.
 */
public class LR_Instance {
    public double features[][];
    public double labels[];
    public String fileName;
    public String delimeter;

    public LR_Instance(String fileName,String delimeter){
        this.fileName=fileName;
        this.delimeter=delimeter;
        this.readFile();
    }
    //default splitString is a blank
    public LR_Instance(String fileName){
        this.fileName=fileName;
        this.delimeter=" ";
        this.readFile();
    }

    public void readFile(){
        try {
            ArrayList<double[]> list=new ArrayList<double[]>();
            //read file stream
            File filename=new File(this.fileName);
            InputStreamReader reader=new InputStreamReader(new FileInputStream(filename));
            BufferedReader br=new BufferedReader(reader);

            if(this.delimeter !=null){delimeter=" ";}
            String line="";
            line=br.readLine();
            int linenum=0;
            int len_tmp=0;
            while(line!=null)
            {
                //features, and last column is label
                String tmp[]=line.split(delimeter);
                len_tmp=tmp.length;
                double value[]=new double[len_tmp];
                for(int i=0;i<len_tmp;i++)
                {
                    value[i]=Double.parseDouble(tmp[i]);
                }
                list.add(value);
                //get the next line
                line=br.readLine();
                linenum++;
            }

            //put data into features and labels
            features=new double[linenum][len_tmp-1];
            labels=new double[linenum];
            Iterator<double[]> dataListIter = list.iterator();
            //double [][] data = new double[list.size()][3];
            int j=0;
            while (dataListIter.hasNext()) {
                double [] tmp = dataListIter.next();
                int i=0;
                for(i=0;i<len_tmp-1;i++)
                {features[j][i]=tmp[i];}
                labels[j]=tmp[i];
            }

        }catch (Exception e){
            e.printStackTrace();
        }
    }
}
