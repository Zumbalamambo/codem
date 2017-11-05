/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.Typeface;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.media.FaceDetector;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Environment;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;
import android.view.View;
import android.widget.ImageView;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.R;
import org.tensorflow.demo.helper.ResizeHelper;

import static org.tensorflow.demo.helper.ResizeHelper.resizeImage;

public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    // These are the settings for the original v1 Inception model. If you want to
    // use a model that's been produced from the TensorFlow for Poets codelab,
    // you'll need to set IMAGE_SIZE = 299, IMAGE_MEAN = 128, IMAGE_STD = 128,
    // INPUT_NAME = "Mul", and OUTPUT_NAME = "final_result".
    // You'll also need to update the MODEL_FILE and LABEL_FILE paths to point to
    // the ones you produced.
    //
    // To use v3 Inception model, strip the DecodeJpeg Op from your retrained
    // model first:
    //
    // python strip_unused.py \
    // --input_graph=<retrained-pb-file> \
    // --output_graph=<your-stripped-pb-file> \
    // --input_node_names="Mul" \
    // --output_node_names="final_result" \
    // --input_binary=true
    public static final int INPUT_SIZE = 160;
    //  private static final int IMAGE_MEAN = 117;
//  private static final float IMAGE_STD = 1;
    public static final String INPUT_NAME = "input";
    public static final String OUTPUT_NAME = "embeddings";

    //  private static final String MODEL_FILE = "file:///android_asset/optimized_20170512-110547.pb";
    public static final String MODEL_FILE = "file:///android_asset/optimized_facenet.pb";
    public static final String LABEL_FILE = Environment.getExternalStorageDirectory() + "/usrID.txt";
    public static final String USRINFO_FILE = Environment.getExternalStorageDirectory() + "/usrINFO.txt";

    private static final boolean SAVE_PREVIEW_BITMAP = false;

    private static final boolean MAINTAIN_ASPECT = true;

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

    private Classifier classifier;

    private Integer sensorOrientation;

    private int previewWidth = 0;
    private int previewHeight = 0;
    private byte[][] yuvBytes;
    private int[] rgbBytes = null;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;

    private Bitmap cropCopyBitmap;
//  private Bitmap bitmap;

    private boolean computing = false;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private ResultsView resultsView;

    private BorderedText borderedText;

    private long lastProcessingTimeMs;
    private int w;
    private int h;
    private Paint paint;

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment;
    }

    @Override
    protected void initView() {
        paint = new Paint();
        paint.setColor(Color.WHITE);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2);
        paint.setTextSize(50);
    }

    private List<Bitmap> facecrop(Bitmap myBitmap) {
// 画中画功能
//        final Canvas canvas = new Canvas(myBitmap);

        List<Bitmap> cutBitmapList = new ArrayList<>();
        // 最多在图片中可找5张人脸
        int numberOfFace = 5;
        // 偏移系数默认值
        float offsetCoefficient = 1.8f;

        int imageWidth = myBitmap.getWidth();
        int imageHeight = myBitmap.getHeight();
        FaceDetector.Face[] myFace = new FaceDetector.Face[numberOfFace];
        FaceDetector myFaceDetect = new FaceDetector(imageWidth, imageHeight, numberOfFace);
        myBitmap = myBitmap.copy(Bitmap.Config.RGB_565, true);
        int numberOfFaceDetected = myFaceDetect.findFaces(myBitmap, myFace);
        Log.w("ww", "number " + numberOfFaceDetected);
        for (int i = 0; i < numberOfFaceDetected; i++) {

            FaceDetector.Face face = myFace[i];
            PointF myMidPoint = new PointF();
            face.getMidPoint(myMidPoint);
            float myEyesDistance = face.eyesDistance();
            float offsetDistance = myEyesDistance * offsetCoefficient;

            int firstPoint_X = (int) (myMidPoint.x - offsetDistance);
            int firstPoint_Y = (int) (myMidPoint.y - offsetDistance);

            int width = (int) (offsetDistance * 2);
            int height = (int) (offsetDistance * 2);
            if (firstPoint_X >= 0 && firstPoint_Y >= 0 && (firstPoint_X + width) < myBitmap.getWidth() && (firstPoint_Y + height) < myBitmap.getHeight()) {
                Bitmap cutBitmap = Bitmap.createBitmap(myBitmap, firstPoint_X, firstPoint_Y, width, height);
//                canvas.drawRect(
//                        (int) (myMidPoint.x - offsetDistance),
//                        (int) (myMidPoint.y - offsetDistance),
//                        (int) (myMidPoint.x + offsetDistance),
//                        (int) (myMidPoint.y + offsetDistance),
//                        paint);
                cutBitmapList.add(cutBitmap);
            } else continue;
        }
//        final Bitmap finalMyBitmap = myBitmap;
//        runOnUiThread(new Runnable() {
//            @Override
//            public void run() {
//                iv_face_detector.setImageBitmap(finalMyBitmap);
//            }
//        });
        return cutBitmapList;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    private static final float TEXT_SIZE_DIP = 10;

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        classifier =
                TensorFlowFaceRecgnizor.create(
                        getAssets(),
                        MODEL_FILE,
                        LABEL_FILE,
                        USRINFO_FILE,
                        INPUT_SIZE,
                        INPUT_NAME,
                        OUTPUT_NAME);

        resultsView = (ResultsView) findViewById(R.id.results);
        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        final Display display = getWindowManager().getDefaultDisplay();
        final int screenOrientation = display.getRotation();

        LOGGER.i("Sensor orientation: %d, Screen orientation: %d", rotation, screenOrientation);

        sensorOrientation = rotation + screenOrientation;

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbBytes = new int[previewWidth * previewHeight];
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        INPUT_SIZE, INPUT_SIZE,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        yuvBytes = new byte[3][];

        addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        renderDebug(canvas);
                    }
                });
    }

    @Override
    public void onImageAvailable(final ImageReader reader) {
        Image image = null;

        try {
            image = reader.acquireLatestImage();


            if (image == null) {
                return;
            }

            if (computing) {
                image.close();
                return;
            }
            computing = true;

            Trace.beginSection("imageAvailable");


            final Plane[] planes = image.getPlanes();//加入人脸检测模块FaceDetector.face
            fillBytes(planes, yuvBytes);

            final int yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();
            ImageUtils.convertYUV420ToARGB8888(
                    yuvBytes[0],
                    yuvBytes[1],
                    yuvBytes[2],
                    previewWidth,
                    previewHeight,
                    yRowStride,
                    uvRowStride,
                    uvPixelStride,
                    rgbBytes);

            image.close();
        } catch (final Exception e) {
            if (image != null) {
                image.close();
            }
            LOGGER.e(e, "Exception!");
            Trace.endSection();
            return;
        }

        rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);//截图转Bitmap

        //bitmap 旋转
        Bitmap tempBitmap = adjustPhotoRotation(rgbFrameBitmap, 90);

        List<Bitmap> faceBitmaplist = facecrop(tempBitmap);
        //裁剪尺寸后的BITMAP 集合
        final List<Bitmap> newfaceBitmaplist = new ArrayList<>();
        if (faceBitmaplist != null) {
            for (int i = 0; i < faceBitmaplist.size(); ++i) {
                Bitmap val = faceBitmaplist.get(i);
                Bitmap newval = ResizeHelper.resizeImage(val, 160, 160);
                newfaceBitmaplist.add(newval);
            }
        } else return;


        final Canvas canvas = new Canvas(croppedBitmap);
//    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {

            ImageUtils.saveBitmap(croppedBitmap);
        }


        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        final long startTime = SystemClock.uptimeMillis();
                        for (Bitmap faceBitmap : newfaceBitmaplist) {
                            final List<Classifier.Recognition> results = classifier.recognizeFace(faceBitmap);
                            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                            resultsView.setResults(results);
                        }

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
//            resultsView.setResults(results);
                        requestRender();
                        computing = false;
                    }
                });

        Trace.endSection();
    }

    public static void reverse(int temp[][]) {
        for (int i = 0; i < temp.length; i++) {
            for (int j = i; j < temp[i].length; j++) {
                int k = temp[i][j];
                temp[i][j] = temp[j][i];
                temp[j][i] = k;
            }
        }
    }

    private Bitmap adjustPhotoRotation(Bitmap bm, final int orientationDegree) {

        Matrix m = new Matrix();
        m.setRotate(orientationDegree, (float) bm.getWidth() / 2, (float) bm.getHeight() / 2);
        float targetX, targetY;
        if (orientationDegree == 90) {
            targetX = bm.getHeight();
            targetY = 0;
        } else {
            targetX = bm.getHeight();
            targetY = bm.getWidth();
        }

        final float[] values = new float[9];
        m.getValues(values);

        float x1 = values[Matrix.MTRANS_X];
        float y1 = values[Matrix.MTRANS_Y];

        m.postTranslate(targetX - x1, targetY - y1);

        Bitmap bm1 = Bitmap.createBitmap(bm.getHeight(), bm.getWidth(), Bitmap.Config.ARGB_8888);
        Paint paint = new Paint();
        Canvas canvas = new Canvas(bm1);
        canvas.drawBitmap(bm, m, paint);

        return bm1;
    }

    @Override
    public void onSetDebug(boolean debug) {
        classifier.enableStatLogging(debug);
    }

    private void renderDebug(final Canvas canvas) {
        if (!isDebug()) {
            return;
        }
        final Bitmap copy = cropCopyBitmap;
        if (copy != null) {
            final Matrix matrix = new Matrix();
            final float scaleFactor = 2;
            matrix.postScale(scaleFactor, scaleFactor);
            matrix.postTranslate(
                    canvas.getWidth() - copy.getWidth() * scaleFactor,
                    canvas.getHeight() - copy.getHeight() * scaleFactor);
            canvas.drawBitmap(copy, matrix, new Paint());

            final Vector<String> lines = new Vector<String>();
            if (classifier != null) {
                String statString = classifier.getStatString();
                String[] statLines = statString.split("\n");
                for (String line : statLines) {
                    lines.add(line);
                }
            }

            lines.add("Frame: " + previewWidth + "x" + previewHeight);
            lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
            lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
            lines.add("Rotation: " + sensorOrientation);
            lines.add("Inference time: " + lastProcessingTimeMs + "ms");

            borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
        }
    }
}
