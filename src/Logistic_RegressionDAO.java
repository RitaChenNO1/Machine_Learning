import java.util.Arrays;

/**
 * Created by chzhenzh on 5/16/2016.
 * Andrew NG's paper
 * https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwi4sdTH0N3MAhXk7YMKHW03AZoQFggfMAA&url=http%3A%2F%2Fcs229.stanford.edu%2Fnotes%2Fcs229-notes1.pdf&usg=AFQjCNFz0r8t9kVFzo04w5vrwlSR0SSUhA&bvm=bv.122129774,d.amc
 */
public class Logistic_RegressionDAO {
    //Gradient factor
   // public double step=0.001;
    //weight of features
    public double []weights;
    public double []likelihood;

    /**
     *
     *Constructor function
     */
    public Logistic_RegressionDAO(){

    }

    /**
     * compute the classification result with input feature values and weight
     * @param x
     * @return
     */
    public double classify(double x[])
    {
        double predict=0.0;
        for(int i=0;i<x.length;i++)
        {
            predict+=x[i]*weights[i];
        }
        return predict;
    }

    /**
     * the sigmoid function
     * @param z
     * @return
     */
    public double sigmoid(double z)
    {
        return 1/(1+Math.exp(-z));
    }

    /**
     * @param max_interation
     * @param instance
     * @param step
     * @return
     *stochastic gradient ascent rule,
     * 1. update the weight
     *h(x) is the predict value of Gradient i
     * wi=wi+step*(h(x) ? y)*xj
     *2. update the likelihood
     * sum( y(i) log h(x(i)) + (1 ? y(i)) log(1 ? h(x(i))) )
      * the previous colunms are features, and the last column is the label(target column)
     */
    public void train( int max_interation,LR_Instance instance, double step){
        double features[][]=instance.features;
        double labels[]=instance.labels;
        weights=new double[features[0].length];
        likelihood=new double[max_interation];
        for(int iter=0;iter<max_interation;iter++) {
            //get every rows data
            for (int i = 0; i < features.length; i++) {
                //get the label,feature values of row i
                double labeli=labels[i];
                double predicti = sigmoid(this.classify(features[i]));
                double error = (labeli-predicti);
                int len_features=features[i].length;
                for(int j=0;j<len_features;j++) {
                    //1. update the weight
                    weights[j] += step * error * features[i][j];
                }
                //2. update the likelihood
                likelihood[iter] += labeli*Math.log(predicti)+(1-labeli)*Math.log(1-predicti);
            }
            System.out.println("iteration: " +iter + " " + Arrays.toString(weights) + " Max likelihood: " + likelihood[iter]);
        }
        //return this.weights;
    }

    public static void main(String args[]){
        LR_Instance data=new LR_Instance("src/logistic_regression_data.txt");
        Logistic_RegressionDAO lr=new Logistic_RegressionDAO();
        lr.train(3000,data,0.001);
    }
}
