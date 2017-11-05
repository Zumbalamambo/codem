package org.tensorflow.demo;

import android.app.Activity;
import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;

public class MainActivity extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    public void input(View view){
        startActivity(new Intent(this, InputActivity.class));}
    public void output_photo(View view){

        startActivity(new Intent(this, OutputPhotoActivity.class));
    }
    public void output_camera(View view){
        startActivity(new Intent(this, ClassifierActivity.class));
    }
}
