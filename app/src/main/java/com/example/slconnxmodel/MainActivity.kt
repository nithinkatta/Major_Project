package com.example.slconnxmodel

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.widget.Button
import android.widget.ImageButton
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.cancel
import java.io.ByteArrayOutputStream
import java.util.Locale

class MainActivity : AppCompatActivity(), SurfaceHolder.Callback {

    private lateinit var surfaceView: SurfaceView
    private lateinit var resultTextView: TextView
    private lateinit var gestureProcessor: GestureProcessor

    private lateinit var surfaceHolder: SurfaceHolder
    private lateinit var headingTextView: TextView
    private lateinit var languageRadioGroup: RadioGroup
    private lateinit var modelRadiogroup: RadioGroup
    private lateinit var audioButton: ImageButton
    private lateinit var resetButton: Button
    private lateinit var captureButton: Button

    private var selectedLanguage = "English"
    private var selectedModel = "model.onnx"

    private val CAMERA_PERMISSION_CODE = 100
    private val TAG = "MainActivity"

    private var camera: android.hardware.Camera? = null
    private lateinit var textToSpeech: TextToSpeech


    // Reusable CoroutineScope
    private val mainScope = CoroutineScope(Dispatchers.Main)
    private var previous_model = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        surfaceView = findViewById(R.id.surfaceView)
        headingTextView = findViewById(R.id.headingTextView)
        languageRadioGroup = findViewById(R.id.languageRadioGroup)
        modelRadiogroup = findViewById(R.id.modelRadiogroup)
        audioButton = findViewById(R.id.audioButton)
        resultTextView = findViewById(R.id.resultTextView)
        resetButton = findViewById(R.id.resetButton)
        captureButton = findViewById(R.id.captureButton)

        // Setup camera preview
        surfaceHolder = surfaceView.holder
        surfaceHolder.addCallback(this)

        // Request camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 101)
        }

        // Language selector listener
        languageRadioGroup.setOnCheckedChangeListener { _, checkedId ->
            selectedLanguage = when (checkedId) {
                R.id.englishRadioButton -> "English"
                R.id.hindiRadioButton -> "Hindi"
                R.id.teluguRadioButton -> "Telugu"
                R.id.chineseRadioButton -> "Chinese"
                else -> "English"
            }

            val sharedPreferences = getSharedPreferences("MyAppPrefs", MODE_PRIVATE)
            val editor = sharedPreferences.edit()
            editor.putString("Language", selectedLanguage)  // Save a string // Save an integer
            editor.apply()
            Toast.makeText(this, "Language selected: $selectedLanguage", Toast.LENGTH_SHORT).show()
        }


        modelRadiogroup.setOnCheckedChangeListener { _, checkedId ->
            val selectedModel = when (checkedId) {
                R.id.aplhamodel -> "random_forest_model_unflatten.onnx"
                R.id.islmodel -> "model.onnx"
                else -> "random_forest_model_unflatten.onnx"
            }

            if (previous_model != selectedModel) {
                previous_model = selectedModel
                // If DemoActivity is already running, notify it of the change
                Log.d("model change trigger",selectedModel)
                GestureProcessor.notifyModelChange(selectedModel)
            }
            val sharedPreferences = getSharedPreferences("ModelTrigger", MODE_PRIVATE)
            val editor = sharedPreferences.edit()
            editor.putString("Model", selectedModel)  // Save a string // Save an integer
            editor.apply()
            if(selectedModel == "random_forest_model_unflatten.onnx")
            {
                Toast.makeText(this, "Model selected: Alphabets", Toast.LENGTH_SHORT).show()
            }
            else
            {
                Toast.makeText(this, "Model selected: Words", Toast.LENGTH_SHORT).show()
            }

        }

        textToSpeech = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = textToSpeech.setLanguage(Locale.US)
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Toast.makeText(this, "Language not supported for TTS", Toast.LENGTH_SHORT).show()
                }
            } else {
                Toast.makeText(this, "Text to Speech initialization failed", Toast.LENGTH_SHORT).show()
            }
        }
        // Audio button click
        audioButton.setOnClickListener {
            val textToSpeak = resultTextView.text.toString()
            if (textToSpeak.isNotBlank() && textToSpeak != "Gesture: None") {
                textToSpeech.speak(textToSpeak, TextToSpeech.QUEUE_FLUSH, null, null)
            } else {
                Toast.makeText(this, "No gesture result to speak", Toast.LENGTH_SHORT).show()
            }
        }

        // Reset button click
        resetButton.setOnClickListener {
            resultTextView.text = "Gesture: None"
            Toast.makeText(this, "Reset done", Toast.LENGTH_SHORT).show()
        }
        // Initialize the gesture processor
        gestureProcessor = GestureProcessor(this)

        // Request camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                CAMERA_PERMISSION_CODE
            )
        } else {
            startCameraPreview()
        }
    }

    private fun startCameraPreview() {
        surfaceView.holder.addCallback(this)
    }

    private fun startCamera(holder: SurfaceHolder, useFrontCamera: Boolean = true, rotation: Int = Surface.ROTATION_0) {
        try {
            // Find the correct camera ID
            var cameraId = 0
            val cameraInfo = android.hardware.Camera.CameraInfo()
            for (i in 0 until android.hardware.Camera.getNumberOfCameras()) {
                android.hardware.Camera.getCameraInfo(i, cameraInfo)
                if (useFrontCamera && cameraInfo.facing == android.hardware.Camera.CameraInfo.CAMERA_FACING_FRONT) {
                    cameraId = i
                    break
                } else if (!useFrontCamera && cameraInfo.facing == android.hardware.Camera.CameraInfo.CAMERA_FACING_BACK) {
                    cameraId = i
                    break
                }
            }

            // Open camera and set parameters
            camera = android.hardware.Camera.open(cameraId)
            camera?.apply {
                setPreviewDisplay(holder)

                // Set display orientation using provided rotation
                val degrees = when (rotation) {
                    Surface.ROTATION_0 -> 0
                    Surface.ROTATION_90 -> 90
                    Surface.ROTATION_180 -> 180
                    Surface.ROTATION_270 -> 270
                    else -> 0
                }
                val displayOrientation = if (cameraInfo.facing == android.hardware.Camera.CameraInfo.CAMERA_FACING_FRONT) {
                    (360 - (cameraInfo.orientation + degrees) % 360) % 360
                } else {
                    (cameraInfo.orientation - degrees + 360) % 360
                }
                setDisplayOrientation(displayOrientation)

                startPreview()

                setPreviewCallback { data, camera ->
                    val parameters = camera.parameters
                    val width = parameters.previewSize.width
                    val height = parameters.previewSize.height
                    val yuvImage = YuvImage(data, ImageFormat.NV21, width, height, null)
                    val out = ByteArrayOutputStream()
                    yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
                    val imageBytes = out.toByteArray()

                    var bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

                    // Mirror image for front camera
                    if (useFrontCamera) {
                        bitmap = Bitmap.createBitmap(bitmap, 0, 0, width, height,
                            Matrix().apply { postScale(-1f, 1f) }, true)
                    }

                    mainScope.launch {
                        val outputResult = gestureProcessor.recognizeGesture(bitmap)
                        resultTextView.text = outputResult
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Camera error: ${e.message}")
            // Removed Toast since it requires context
        }
    }

    private fun stopCamera() {
        camera?.apply {
            setPreviewCallback(null)
            stopPreview()
            release()
        }
        camera = null
    }

    // SurfaceHolder.Callback implementations
    override fun surfaceCreated(holder: SurfaceHolder) {
        startCamera(holder)
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        // No action needed here
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        stopCamera()
    }

    // Handle permission result
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCameraPreview()
            } else {
                Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // Stop camera on pause to save resources
    override fun onPause() {
        super.onPause()
        stopCamera()
    }

    // Restart camera on resume
    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCameraPreview()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        mainScope.cancel() // Clean up the coroutine scope
    }
}
