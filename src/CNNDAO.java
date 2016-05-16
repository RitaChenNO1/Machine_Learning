/**
 * Created by chzhenzh on 5/9/2016.
 */

import java.util.Arrays;
import java.util.Random;

/**
 * implement conventional Neural network with java code
 * reference bellow web pages
 * 1.
 * http://wenku.baidu.com/link?url=7-eey3LfTm1eu1wfwPguMuXVLZRU_kGZEtwL78DWMEkbSfGGskP3SmVE0h3aOZou6FQz8Aqwa3RTjsLjbPYDud5fdW3WwicjA8-Z609c0ca
 * the formulas explaination on page 9, page 10
 * 2.
 * http://baike.baidu.com/link?url=-MpDgZajmCxs3EJKKdw4OCWQImyIZJzH6KMhtwvsXMqajHajCAANEFW6GKCW6gfq2m6RovgnA_CESaacGNaHka
 * self learning model
 * ?Wij(n+1)= h ×?i×Oj+a×?Wij(n)
 * h -learning factor??i-the error of output node i?Oj-output value of output node j?a-move factor?
 * 3.
 * http://blog.csdn.net/zdy0_2004/article/details/50685294
 * java source code reference
 */
public class CNNDAO {
    /**
     *   definition for layer, weight,weight?, and error of layer's node
     */

    //layer, node id, and node value
    public double[][] layer;
    //layer, node id, next layer node id, and weight value
    public double [][][] layer_weight;
    //layer, node id, next layer node id, and weight delta value
    public double [][][] layer_weight_delta;
    //layer, node id, next layer node id, and weight delta value, with learning factor and move factor
    public double [][][] layer_weight_delta_selfLearn;
    //layer, node id, and node error value
    public double[][] layer_Error;
    //learning factor h
    public double h;
    //move factor a
    public double a;

    /**
     * initial the value of all nodes' weight as random value 0-1;
     * arg 1, the input nodes numbers,nodes numbers of every hidden layers, output nodes numbers
     * arg 2, learning factor h
     * arg 3, move factor
     */
    public  CNNDAO(int nodes[], double h, double a)
    {
        //1. inital all arrays' layer of neural network
        this.h=h;
        this.a=a;
        int layer_length=nodes.length;
        layer=new double[layer_length][];
        layer_weight=new double[layer_length][][];
        layer_weight_delta=new double[layer_length][][];
        layer_weight_delta_selfLearn=new double[layer_length][][];
        layer_Error=new double[layer_length][];
        Random random=new Random();
        //input layer id as i, output id as o, and layer id as l
        for(int l=0;l<layer_length;l++)
        {
            //get the #nodes of every layer, # means numbers
            //inital all nodes of corresponding layer
            int layer_nodes=nodes[l];
            layer[l]=new double[layer_nodes];
            layer_Error[l]=new double[layer_nodes];
            //initial weight with random value 0-1, from layer input to hidden,hidden to hidden,hidden to output
            for(int i=0;i<layer_nodes;i++)
            {
                //#nodes of next layers

                if(l+1<layer_length)
                {
                   // System.out.println("l:"+l);
                    int next_layer_nodes=nodes[l+1];
                    //need to add 1 node for weight, since the intercept is there, except the output layer
                    layer_weight[l] = new double[layer_nodes + 1][next_layer_nodes];
                    layer_weight_delta[l] = new double[layer_nodes + 1][next_layer_nodes];
                    layer_weight_delta_selfLearn[l] = new double[layer_nodes + 1][next_layer_nodes];
                    for(int o=0;o<next_layer_nodes;o++)
                    {
                        //initial the value of all nodes' weight as random value 0-1;
                      //  System.out.println("l:"+l+" i:"+i+" o:"+o);
                        layer_weight[l][i][o] = random.nextDouble();
                    }
                }

            }
        }
    }

    /**
     * Step by step forward to calculate the output
     *  compute the value from layer to layer,value(layer l-1)1*w1+value(layer l-1)2*w2+...
     *  arg 1, the input nodes value
     */

    public double[] info_input_hidden_output(double []inputnodes)
    {
        int layer_length=layer.length;
        for(int l=1;l<layer_length;l++)
        {
            int layer_lpevious_nodes=layer[l-1].length;
            int layer_l_nodes=layer[l].length;
            //the first layer is input layer, is equal to inputnodes
            if(l==1)
            {
                for (int o = 0; o < layer_lpevious_nodes; o++) {
                    layer[l][o] = inputnodes[o];
                }
            }
            //set the value of layer l, node output;
            for (int o = 0; o < layer_l_nodes; o++) {
                //current layer has output node, it need to summary previous layer's *weight
                //initial is the intercept???????
                //double tmpY=layer_weight[l-1][layer_l_nodes-1][o];
                double tmpY=0;
                for (int i = 0; i < layer_lpevious_nodes; i++) {
                    tmpY+=layer[l-1][i]*layer_weight[l-1][i][o];
                }
                //layer[l][i]=tmpY;
                //sigmoid transformation [0,1], it could be tang[-1,1] or others [multiple classes]
                layer[l][o]=1/(1+Math.exp(-tmpY));
            }
        }
        return layer[layer_length-1];
    }

    /**
     * layer reverse calculation error and modify the weights
     * compute the error, and deliver the error from output-->hidden-->input
     *
     */
    public void reverse_error_output_hidden_input(double []target)
    {
        //1. compute the error of last layer, the output layer
        int l=layer.length-1;
        for(int o=0; o<layer[l].length ;o++)
        {
            layer_Error[l][o]=layer[l-1][o]*(1-layer[l-1][o])*(target[o]-layer[l][o]);
        }
        System.out.println("layer_Error output layer:");
        System.out.println(Arrays.toString(layer_Error[l]));
        //move to the last hidden layer
        l--;
        while(l>0)
        {
            for(int i=0;i<layer[l].length;i++)
            {
                double Err=0.0;
            //step 1. compute weight delta  xi*error(Yj), x1*error(Y1)+x1*erro(Y2)+...
            // 2.compute weight delta with move factor and learning factor,   h ×Oj×error(i)+a× delta(Wij(n)), delta(wij(n)) from step 1
            // 3. compute weight, old weight+weight delta self learning
                for(int o=0;o<layer[l+1].length;o++)
                {
                    //System.out.println("l:"+l+" i:"+i+" o:"+o);
                    layer_weight_delta[l][i][o]=layer[l][i]*layer_Error[l+1][o];
                    layer_weight_delta_selfLearn[l][i][o]=h*layer[l][i]*layer_Error[l+1][o]+a*layer_weight_delta[l][i][o];
                    layer_weight[l][i][o]+=layer_weight_delta_selfLearn[l][i][o];
                    //add intercept b;
                    //4. compute the delta, self learning delta, weight of intercept
                    if(i==layer[l].length-1){
                        layer_weight_delta[l][i+1][o]=layer_Error[l+1][o];
                        layer_weight_delta_selfLearn[l][i+1][o]=h*layer_Error[l+1][o]+a*layer_weight_delta[l][i][o];
                        layer_weight[l][i+1][o]+=layer_weight_delta_selfLearn[l][i+1][o];
                    }
                    //comput Err(yj)
                    if(l>0)
                    {
                        Err+=layer_weight[l][i][o]*layer_Error[l+1][o];
                    }else{
                        Err=0.0;
                    }

                }
                System.out.println("layer_weight:");
                System.out.println("l:"+l+" i:"+i+":"+Arrays.toString(layer_weight[l][i]));
                //5. compute error, yj*(1-yj)*Err(yj); Err(yj)= wj1*error(Z1)+yj2*error(Z2)+...
                layer_Error[l][i]=layer[l][i]*(1-layer[l][i])*Err;
        }
            System.out.println("layer_Error:");
            System.out.println(Arrays.toString(layer_Error[l]));
            //reverse to previous layer
            l--;

        }



    }

    public void train(double []inputnodes,double []target )
    {
        info_input_hidden_output(inputnodes);
        reverse_error_output_hidden_input(target);
    }

}
