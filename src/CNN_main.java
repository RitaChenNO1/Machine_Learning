import java.util.Arrays;

/**
 * Created by chzhenzh on 5/9/2016.
 */
public class CNN_main {

    public static void main(String args[]){
        CNNDAO cnn=new CNNDAO(new int[]{2,2,2},0.15,0.8);
        //sample data
        double[][] data = new double[][]{{1,2},{2,2},{1,1},{2,1}};
        double[][] target=new double[][]{{1,0},{0,1},{0,1},{1,0}};
        for(int tid=0;tid<1;tid++){
            for(int i=0;i<data.length;i++) {
                cnn.train(data[i], target[i]);
            }
        }

        for(int i=0;i<data.length;i++) {
            double [] output=cnn.info_input_hidden_output(data[i]);
            System.out.println(Arrays.toString(data[i])+" : "+Arrays.toString(output));
        }

        double[] val=new double[]{3,1};
        double[] output=cnn.info_input_hidden_output(val);
        System.out.println(Arrays.toString(val)+" : "+Arrays.toString(output));
    }
}
