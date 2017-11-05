package org.tensorflow.demo;

import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.provider.MediaStore;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.Toast;

import org.tensorflow.demo.helper.DetectFaceHelper;
import org.tensorflow.demo.helper.ResizeHelper;

import java.util.List;

import static org.tensorflow.demo.ClassifierActivity.INPUT_NAME;
import static org.tensorflow.demo.ClassifierActivity.INPUT_SIZE;
import static org.tensorflow.demo.ClassifierActivity.LABEL_FILE;
import static org.tensorflow.demo.ClassifierActivity.MODEL_FILE;
import static org.tensorflow.demo.ClassifierActivity.OUTPUT_NAME;
import static org.tensorflow.demo.ClassifierActivity.USRINFO_FILE;

public class OutputPhotoActivity extends Activity {

    private static final String TAG = "OutputPhotoActivity";
    private static final int RESULT_LOAD_IMAGE = 9999;
    private ImageView imageView;
    private DetectFaceHelper detectFaceHelper = new DetectFaceHelper();
    private ResultsView resultsView;
    private Classifier classifier;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_output_photo);

        imageView = (ImageView) findViewById(R.id.imgView);
        resultsView = (ResultsView) findViewById(R.id.results);


        classifier =
                TensorFlowFaceRecgnizor.create(
                        getAssets(),
                        MODEL_FILE,
                        LABEL_FILE,
                        USRINFO_FILE,
                        INPUT_SIZE,
                        INPUT_NAME,
                        OUTPUT_NAME);

        Intent i = new Intent(
                Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);

        startActivityForResult(i, RESULT_LOAD_IMAGE);

    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == RESULT_LOAD_IMAGE && resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();
            String[] filePathColumn = {MediaStore.Images.Media.DATA};

            Cursor cursor = getContentResolver().query(selectedImage,
                    filePathColumn, null, null, null);
            cursor.moveToFirst();

            int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
            String picturePath = cursor.getString(columnIndex);
            cursor.close();


            Bitmap recsBitmap = BitmapFactory.decodeFile(picturePath);
            Bitmap bitmap = recsBitmap.copy(Bitmap.Config.RGB_565, true);

            List<Bitmap> facecrop = detectFaceHelper.facecrop(bitmap);
            Log.w(TAG, "recsBitmap: " + recsBitmap.getRowBytes());
            Log.w(TAG, "onActivityResult: " + facecrop.size()   + "," + bitmap.getRowBytes());
            imageView.setImageBitmap(bitmap);

        if (facecrop.size() == 0 || facecrop.get(0) == null){
            Toast.makeText(this, "当前相片检测不到人脸", Toast.LENGTH_SHORT).show();
        }else{

            Bitmap newval = ResizeHelper.resizeImage(facecrop.get(0), 160, 160);
            List<Classifier.Recognition> results = classifier.recognizeFace(newval);
            resultsView.setResults(results);
        }

            // String picturePath contains the path of selected Image
        }else{
            finish();
        }
    }
}
