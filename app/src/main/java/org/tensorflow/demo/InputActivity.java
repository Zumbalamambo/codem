package org.tensorflow.demo;

import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Toast;

import org.tensorflow.demo.helper.DetectFaceHelper;
import org.tensorflow.demo.helper.ResizeHelper;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import static org.tensorflow.demo.ClassifierActivity.INPUT_NAME;
import static org.tensorflow.demo.ClassifierActivity.INPUT_SIZE;
import static org.tensorflow.demo.ClassifierActivity.LABEL_FILE;
import static org.tensorflow.demo.ClassifierActivity.MODEL_FILE;
import static org.tensorflow.demo.ClassifierActivity.OUTPUT_NAME;
import static org.tensorflow.demo.ClassifierActivity.USRINFO_FILE;

public class InputActivity extends Activity {

    private static final String TAG = "OutputPhotoActivity";
    private static final int RESULT_LOAD_IMAGE = 9999;
    private ImageView imageView;
    private DetectFaceHelper detectFaceHelper = new DetectFaceHelper();
//    private ResultsView resultsView;
    private Classifier classifier;
    private EditText et;
    private float[] results;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_input);

        imageView = (ImageView) findViewById(R.id.imgView);
        et = (EditText) findViewById(R.id.et);
//        resultsView = (ResultsView) findViewById(R.id.results);


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
                results = classifier.enteringFace(newval);
            }

            // String picturePath contains the path of selected Image
        }else{
            finish();
        }
    }

    public void confirm(View view){
        String name = et.getText().toString().trim();
//        StringBuffer temp = results

        StringBuffer buffer = new StringBuffer();
        for (float ff : results)
            buffer.append(ff + " ");

        try {
            FileWriter idWriter = new FileWriter(LABEL_FILE, true);
            BufferedWriter bufferIdWriter = new BufferedWriter(idWriter);
            bufferIdWriter.newLine();
            bufferIdWriter.write(name);
            bufferIdWriter.flush();
            bufferIdWriter.close();
            idWriter.close();

            FileWriter infoWriter = new FileWriter(USRINFO_FILE, true);
            BufferedWriter bufferInfoWriter = new BufferedWriter(infoWriter);
            bufferInfoWriter.newLine();
            bufferInfoWriter.write(buffer.toString());
            bufferInfoWriter.flush();
            bufferInfoWriter.close();
            infoWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
            Log.w(TAG, "confirm: "  + e.getMessage());
        }
    }
}
