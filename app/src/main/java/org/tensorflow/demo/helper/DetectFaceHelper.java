package org.tensorflow.demo.helper;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PointF;
import android.media.FaceDetector;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by 李振强 on 2017/9/12.
 */

public class DetectFaceHelper {


    private Paint paint;

    public List<Bitmap> facecrop(Bitmap myBitmap) {
        List<Bitmap> cutBitmapList = new ArrayList<>();
        // 设成了1，只找出一个可信度最高的人脸
        int numberOfFace = 1;
        // 偏移系数默认值
        float offsetCoefficient = 1.7f;

        int imageWidth = myBitmap.getWidth();
        int imageHeight = myBitmap.getHeight();
        FaceDetector.Face[] myFace = new FaceDetector.Face[numberOfFace];
        FaceDetector myFaceDetect = new FaceDetector(imageWidth, imageHeight, numberOfFace);
        int numberOfFaceDetected = myFaceDetect.findFaces(myBitmap, myFace);
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
            Bitmap cutBitmap = null;
            if (width + firstPoint_X > imageWidth || height + firstPoint_Y > imageHeight) {
                cutBitmap = Bitmap.createBitmap(myBitmap);
            } else {
                cutBitmap = Bitmap.createBitmap(myBitmap, firstPoint_X, firstPoint_Y, width, height);

                // 把人脸标出来

                Canvas canvas = new Canvas(myBitmap);

                Paint myPaint = new Paint();
                myPaint.setColor(Color.GREEN);
                myPaint.setStyle(Paint.Style.STROKE);
                myPaint.setStrokeWidth(3);

                canvas.drawRect(
                        (int) (myMidPoint.x - offsetDistance),
                        (int) (myMidPoint.y - offsetDistance),
                        (int) (myMidPoint.x + offsetDistance),
                        (int) (myMidPoint.y + offsetDistance),
                        myPaint);
            }


            cutBitmapList.add(cutBitmap);
        }
        return cutBitmapList;
    }

}
