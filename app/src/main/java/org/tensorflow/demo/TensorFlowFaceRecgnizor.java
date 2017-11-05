/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Trace;
import android.util.Log;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

/** A classifier specialized to label images using TensorFlow. */
public class TensorFlowFaceRecgnizor implements Classifier {
    private static final String TAG = "TensorFloFaceRecgnizor";

    // Only return this many results with at least this confidence.
    private static final int MAX_RESULTS = 3;
    private static final float THRESHOLD = 1.8f;

    // Config values.
    private String inputName;
    private String outputName;
    private int inputSize;
//    private int imageMean;
//    private float imageStd;

    // Pre-allocated buffers.
    private Vector<String> userID =new Vector<String>();
    private float[][] linearray;
    private int[] intValues;
    private float[] floatValues;
    private float[] outputs;
    private String[] outputNames;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    private TensorFlowFaceRecgnizor() {}

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     * @param usrFilename the filepath of embeddings file
     * @param inputSize The input size. A square image of inputSize x inputSize is assumed.
//     * @param imageMean The assumed mean of the image values.
//     * @param imageStd The assumed std of the image values.
     * @param inputName The label of the image input node.
     * @param outputName The label of the output node.
     * @throws IOException
     */
    public static Classifier create(
            AssetManager assetManager,
            String modelFilename,
            String labelFilename,
            String usrFilename,
            int inputSize,
            String inputName,
            String outputName) {
        TensorFlowFaceRecgnizor c = new TensorFlowFaceRecgnizor();
        c.inputName = inputName;
        c.outputName = outputName;
//        String actualusrFilename = usrFilename.split("file:///android_asset/")[1];
//        Log.i(TAG, "Reading labels from: " + actualusrFilename);
        BufferedReader br = null;

//        float[][] array = new float[5][];

        try {
            br = new BufferedReader(new InputStreamReader(new FileInputStream(usrFilename)));
//            String line;
            String[] sp;
            List<String> list = new ArrayList<String>();
//            float[] lineArray;
//            float[][] array = null; // 2行
//            int ind = 0;
            String line = null;
            while ((line = br.readLine()) != null) {
                if (!line.startsWith("#"))
                list.add(line);}
            float[][] array = new float[list.size()][128];
//                float[][] array = new float[list.size()][128];
//                int i;
                for(int i=0,ls=list.size();i<ls;++i){
                sp = list.get(i).split(" ");
//                lineArray = new float[sp.length];
//      float[][]array =null;

                    for(int j=0,l=sp.length;j<l;++j) {
                     array[i][j] = Float.parseFloat(sp[j]);

                }

            br.close();
            c.linearray=array;
        } }
        catch (IOException e) {
            throw new RuntimeException("Problem reading INFO file!" , e);
        }
//        c.linearray=array;
//    String actualusrFilename = usrFilename.split("file:///android_asset/")[1];
////    FileReader fr=new FileReader();
//    //可以换成工程目录下的其他文本文件
//    BufferedReader br=new BufferedReader(new InputStreamReader(assetManager.open(actualusrFilename)));
//    String line = null;
//    float[][] array = null; // 2行
//    int ind = 0;
//    String[] sp;
//    float[] lineArray=null;
//    while((line=br.readLine())!=null){
//      sp = line.split("   "); //将mat文件复制到txt后数据间默认是一个tab的距离
//      lineArray = new float[sp.length];
////      float[][]array =null;
//
//      for(int j=0,l=sp.length;j<l;++j){
//        lineArray[j] = Float.parseFloat(sp[j]);
//        //System.out.println(lineArray[j]);
////         System.out.print(lineArray[j]+" ");//这种输出方式使输出数据后不换行
//        }
////        System.out.println("\n");
//        //  System.out.println(line);
//        array[ind++] = lineArray;
//
//      }

//           array[ind++] = lineArray;

        // Read the label names into memory.
        // TODO(andrewharp): make this handle non-assets.
//    String actualFilename = labelFilename.split("file:///android_asset/")[1];
//    Log.i(TAG, "Reading labels from: " + actualFilename);
    BufferedReader lbr = null;
    try {
        lbr = new BufferedReader(new InputStreamReader(new FileInputStream(labelFilename), Charset.forName("utf-8")));
        String userline=null;
        while ((userline = lbr.readLine()) != null) {
            if(!userline.startsWith("#"))
//                IDlist=IDlist.add(userline);
            c.userID.add(userline);
        }//用户信息读取进入userID
        lbr.close();
    } catch (IOException e) {
        throw new RuntimeException("Problem reading ID file!" , e);
    }
//    public float measure(float[] truth,float[] prediction) {
//        if (truth.length != prediction.length) {
//            throw new IllegalArgumentException(String.format("The vector sizes don't match: %d != %d.", truth.length, prediction.length));
//        }
//
//        int n = truth.length;
//        double rss = 0.0;
//        for (int i = 0; i < n; i++) {
//            rss += (truth[i] - prediction[i])*(truth[i] - prediction[i]);
//        }
//
//
//    }
        c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

        // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
        final Operation operation = c.inferenceInterface.graphOperation(outputName);
        final int numClasses = (int) operation.output(0).shape().size(1);
        Log.i(TAG, "Read " + c.userID.size() + " labels, output layer size is " + numClasses);

        // Ideally, inputSize could have been retrieved from the shape of the input operation.  Alas,
        // the placeholder node for input in the graphdef typically used does not specify a shape, so it
        // must be passed in as a parameter.
        c.inputSize = inputSize;
//        c.imageMean = imageMean;
//        c.imageStd = imageStd;

        // Pre-allocate buffers.
        c.outputNames = new String[] {outputName};
        c.intValues = new int[inputSize * inputSize];
        c.floatValues = new float[inputSize * inputSize * 3];
        c.outputs = new float[numClasses];

        return c;
    }

    public float[] measure(float[][] truth,float[] prediction) {
        int ind;

        float[] rss=new float[truth.length];
        for ( ind=0;ind<truth.length;ind++)
        {
        if (truth[0].length != prediction.length) {
            throw new IllegalArgumentException(String.format("The vector sizes don't match: %d != %d.", truth.length, prediction.length));
        }

        int n = truth[0].length;
//        float[] rss = null;
        for (int i = 0; i < n; i++) {
            rss[ind] += (truth[ind][i] - prediction[i])*(truth[ind][i] - prediction[i]);
        }
//    return rss;

    }return rss;
    }


    //    @Override
    public List<Recognition> recognizeFace(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = ((val >> 16) & 0xFF) ;
            floatValues[i * 3 + 1] = ((val >> 8) & 0xFF) ;
            floatValues[i * 3 + 2] = (val & 0xFF) ;
        }
        Trace.endSection();

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3);
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputName, outputs);
        Trace.endSection();


        // Find the best classifications.
        PriorityQueue<Recognition> pq =
                new PriorityQueue<Recognition>(
                        3,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(-rhs.getConfidence(), -lhs.getConfidence());
                            }
                        });

        for (int i = 0; i < linearray.length; ++i) {
            float score[]=measure(linearray,outputs);
            if (score[i] !=0) {
                pq.add(
                        new Recognition(
                                "" + i, score[i] <THRESHOLD ? userID.get(i) : "unknown", score[i], null));
            }
        }
        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        Trace.endSection(); // "recognizeImage"
        return recognitions;
    }

    @Override
    public float[] enteringFace(Bitmap bitmap) {

        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = ((val >> 16) & 0xFF) ;
            floatValues[i * 3 + 1] = ((val >> 8) & 0xFF) ;
            floatValues[i * 3 + 2] = (val & 0xFF) ;
        }
        Trace.endSection();

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3);
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputName, outputs);
        Trace.endSection();

        return outputs;
    }

    @Override
    public void enableStatLogging(boolean logStats) {
        this.logStats = logStats;
    }

    @Override
    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }
}
