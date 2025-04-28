package com.example.slconnxmodel

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import ai.onnxruntime.*
import android.app.Activity
import android.content.BroadcastReceiver
import android.content.Context.MODE_PRIVATE
import android.content.Intent
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.mediapipe.formats.proto.LandmarkProto
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class GestureProcessor(private val context: Context) : AppCompatActivity() {

    private lateinit var env: OrtEnvironment
    private lateinit var session: OrtSession
    private lateinit var handLandmarker: HandLandmarker

    private val word_labels = listOf(
        "call", "you", "are", "four", "how", "I", "ok", "you", "hi", "peace",
        "please", "love", "stop", "help", "four", "sorry","wait","help"
    )
    private val aplhabets_labels  = listOf(
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "V", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "K", "Q", "nothing"
    )

    init {
        instance = this
        loadModel()
        setupHandLandmarker()
    }

    // Load the ONNX model from assets
    private fun loadModel() {
        try {
            val modelFile = assetFilePath(context, "random_forest_model_unflatten.onnx")
            env = OrtEnvironment.getEnvironment()
            session = env.createSession(modelFile, OrtSession.SessionOptions())
            Log.d("GestureProcessor", "ONNX Model loaded successfully!")
        } catch (e: Exception) {
            Log.e("GestureProcessor", "Error loading ONNX model: ${e.message}", e)
        }
    }

    // Initialize MediaPipe HandLandmarker
    private fun setupHandLandmarker() {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .build()

            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setNumHands(1)
                .build()

            handLandmarker = HandLandmarker.createFromOptions(context, options)
            Log.d("GestureProcessor", "HandLandmarker initialized successfully!")
        } catch (e: Exception) {
            Log.e("GestureProcessor", "Error initializing HandLandmarker: ${e.message}", e)
        }
    }

    // Helper function to get model file path from assets
    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) return file.absolutePath

        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
        return file.absolutePath
    }

    // Preprocess the cropped hand image
    private fun preprocessFrame(frame: Bitmap): FloatBuffer {
        val resizedBitmap = Bitmap.createScaledBitmap(frame, 224, 224, true)
        val rgbBitmap = resizedBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val floatBuffer = FloatBuffer.allocate(3 * 224 * 224)

        TensorImageUtils.bitmapToFloatBuffer(
            rgbBitmap,
            0, 0, 224, 224,
            floatArrayOf(0.0f, 0.0f, 0.0f),
            floatArrayOf(1.0f, 1.0f, 1.0f),
            floatBuffer,
            0
        )

        floatBuffer.rewind()

        Log.d("alphabets model",floatBuffer.toString())
        return floatBuffer
    }

    // Postprocess model output
    private fun postprocessOutput_words_model(output: FloatArray): Pair<String, Float> {
        val maxIndex = output.indices.maxByOrNull { output[it] } ?: -1
        val gesture = if (maxIndex in word_labels.indices) word_labels[maxIndex] else "Unknown Gesture"
        val confidence = output.getOrElse(maxIndex) { 0F }
        Log.d("gesture:",gesture)
        Log.d("Confidence",confidence.toString())
        return Pair(gesture, confidence)
    }
    private fun postprocessOutput(output: LongArray): Pair<String, Long> {
        val maxIndex = output.indices.maxByOrNull { output[it] } ?: -1
//        var gesture = if (maxIndex in aplhabets_labels.indices) aplhabets_labels[maxIndex] else "Unknown Gesture"
        val confidence = output.getOrElse(maxIndex) { 0L }
        val gesture = aplhabets_labels[confidence.toInt()]
        Log.d("gesture:",gesture)
        Log.d("Confidence",confidence.toString())
        return Pair(gesture, confidence)
    }



    fun extractAdvancedFeatures(landmarks: FloatArray): FloatArray {
        val numLandmarks = 21
        val numCoords = 3

        // Reconstruct landmarkList
//        val landmarkList = Array(numLandmarks) { i ->
//            floatArrayOf(
//                landmarks[i * 3],
//                landmarks[i * 3 + 1],
//                landmarks[i * 3 + 2]
//            )
//        }
        val landmarkList = mutableListOf<FloatArray>()
        for (i in 0 until landmarks.size step 3) {
            landmarkList.add(floatArrayOf(landmarks[i], landmarks[i + 1], landmarks[i+2]))
        }

        Log.d("landmarks",landmarks.toString())
        Log.d("landmarks",landmarkList.size.toString())

//        landmarks.forEach { println(it.toString()) }

        val deltas = mutableListOf<Float>()
        for (i in landmarkList.indices) {
            for (j in i + 1 until landmarkList.size) {
                val dx = landmarkList[i][0] - landmarkList[j][0]
                val dy = landmarkList[i][1] - landmarkList[j][1]
                val dz = landmarkList[i][2] - landmarkList[j][2]
                val dist = sqrt(dx * dx + dy * dy + dz * dz)
                deltas.add(dist)
            }
        }



        // Palm center (mean)
        val palmCenter = FloatArray(3)
        for (lm in landmarkList) {
            palmCenter[0] += lm[0]
            palmCenter[1] += lm[1]
            palmCenter[2] += lm[2]
        }
//        palmCenter[0] /= numLandmarks
//        palmCenter[1] /= numLandmarks
//        palmCenter[2] /= numLandmarks

        // Orientation features (landmark - center)
        val orientation = mutableListOf<Float>()
        for (lm in landmarkList) {
            orientation.add(lm[0] - palmCenter[0])
            orientation.add(lm[1] - palmCenter[1])
            orientation.add(lm[2] - palmCenter[2])
        }

        // Final feature vector
        val totalSize = landmarks.size + deltas.size + orientation.size // 63 + 1890 + 63 = 2016
        val features = FloatArray(totalSize)
        var index = 0
        landmarks.forEach { features[index++] = it }
        deltas.forEach { features[index++] = it }
        orientation.forEach { features[index++] = it }
        Log.d("landmards",landmarks.size.toString())
        Log.d("deltas",deltas.size.toString())
        Log.d("orientation",orientation.size.toString())
        Log.d("feature",features.size.toString())
        return features
    }



    // Make this function suspend
    suspend fun processGeminiResponse(prompt: String): String {
        return suspendCancellableCoroutine { continuation ->
            GeminiApiHelper.sendPrompt(prompt) { response ->
                if (response != null) {
                    continuation.resume(response, null)
                } else {
                    Log.e("GeminiResponse", "Failed to get response from Gemini AI")
                    continuation.resume("Failed to get response from Gemini AI", null)
                }
            }
        }
    }


    // Make this function suspend as well
    private val sentenceBuilder = mutableListOf<String>()
    private var previous_response = ""
    private var previous_gesture = ""
    private var flag = true
    private var current_model = ""

    private fun loadModelIfNeeded() {
        val sharedPreferences = context.getSharedPreferences("ModelTrigger", Context.MODE_PRIVATE)
        val selectedModelPath = sharedPreferences.getString("Model", "model.onnx")

        try {

            val modelFile = assetFilePath(context, selectedModelPath.toString())
            env = OrtEnvironment.getEnvironment()
            session = env.createSession(modelFile, OrtSession.SessionOptions())
            current_model = selectedModelPath.toString()

            Log.d("Selected Model",selectedModelPath.toString())
            Log.d("GestureProcessor", "ONNX Model loaded successfully!")
        } catch (e: Exception) {
            Log.e("GestureProcessor", "Error loading ONNX model: ${e.message}", e)
        }
    }

    companion object {
        private var instance: GestureProcessor? = null
            set(value) {
                Log.d("GestureProcessor", "Instance ${if (value != null) "set" else "cleared"}")
                field = value
            }

        fun notifyModelChange(model: String) {
            Log.d("GestureProcessor", "Notify called, instance=${instance != null}")
            instance?.apply {
                Log.d("GestureProcessor", "Processing model change to $model")
                // ... existing code ...
                val modelFile = assetFilePath(context, model)
                env = OrtEnvironment.getEnvironment()
                session = env.createSession(modelFile, OrtSession.SessionOptions())
                current_model = model
                Log.d("notifyModelChange trigger",model)
            }
        }
    }
    override fun onStart() {
        super.onStart()
        instance = this // Set instance when activity starts
    }

    override fun onStop() {
        super.onStop()
        instance = null // Clear instance when activity stops
    }


    suspend fun recognizeGesture(bitmap: Bitmap): String {
        try {
            // Step 1: Detect hand landmarks
//            loadModelIfNeeded()

            val mpImage = BitmapImageBuilder(bitmap).build()
            val result = handLandmarker.detect(mpImage)
            Log.d("model change",current_model)
            if( current_model == "model.onnx")
            {
                if (result.landmarks().isEmpty()) {
                    Log.d("GestureProcessor", "No hand detected.")

                    val currentTime = System.currentTimeMillis()
                    val geminiRequestCooldown = 5000L // 5 seconds

                    // Store this in a property outside the function to persist between calls
                    if ((currentTime - lastGeminiRequestTime >= geminiRequestCooldown) && flag == true) {
                        lastGeminiRequestTime = currentTime

                        val sentence = sentenceBuilder.joinToString(" ")
                        val words = sentence.split(" ")

                        val sharedPreferences =
                            context.getSharedPreferences("MyAppPrefs", MODE_PRIVATE)
                        val language = sharedPreferences.getString("Language", "English")

                        val prompt=
                            "role: You are a professional sign language recogniser and sentence generator," +
                            "Task: You need to generate a meaningful sentence based on the given input data," +
                            "Assistant:  without any explanation give the exact output, avoid duplicate words from the data given, output must be a casual dialogue" +
                            "language: " + language +
                            "Data: "+ sentence

                        Log.d("sentence", sentence)
                        val response = processGeminiResponse(prompt)
                        sentenceBuilder.clear()
                        Log.d("GeminiResponse", response)
                        previous_response = response
                        flag = false
//                        return Pair(response, "")
                        return response
                    } else {
//                    Log.d("GeminiResponse", "Skipping Gemini API call to avoid flooding")
//                        return Pair(previous_response, "")
                        return previous_response
                    }
                }

                flag = true

                // Step 2: Get bounding box around the hand
                val handLandmarks = result.landmarks()[0]
                val boundingBox = getBoundingBox(handLandmarks, bitmap.width, bitmap.height)

                // Step 3: Crop the hand region
                val croppedBitmap = Bitmap.createBitmap(
                    bitmap,
                    boundingBox.left,
                    boundingBox.top,
                    boundingBox.width(),
                    boundingBox.height()
                )

                // Step 4: Preprocess the cropped hand image
                Log.d("words model croppedBitmap", croppedBitmap.toString())
                val floatBuffer = preprocessFrame(croppedBitmap)

                // Step 5: Create input tensor with correct shape
                val shape = longArrayOf(1, 3, 224, 224)
//            val shape = longArrayOf(1, 2079)
                val inputTensor = OnnxTensor.createTensor(env, floatBuffer, shape)


                // Step 6: Run inference
                val resultMap =
                    session.run(mapOf(session.inputNames.iterator().next() to inputTensor))

                val outputTensor = resultMap[0].value as Array<FloatArray>
                val outputArray = outputTensor[0]

                // Step 7: Postprocess the output
                val output_result = postprocessOutput_words_model(outputArray)
                var detected_gesture = ""

                output_result.let {
                    detected_gesture = it.first
                }
                detected_gesture = output_result.first

                if (detected_gesture != previous_gesture)
                {
                    sentenceBuilder.add(detected_gesture)
                }
                previous_gesture = detected_gesture


                return output_result.first
            }



            else {
                if (result.landmarks().isEmpty()) {
                    Log.d("GestureProcessor", "No hand detected.")

                    val currentTime = System.currentTimeMillis()
                    val geminiRequestCooldown = 5000L // 5 seconds

                    // Store this in a property outside the function to persist between calls
                    if ((currentTime - lastGeminiRequestTime >= geminiRequestCooldown) && flag == true) {
                        lastGeminiRequestTime = currentTime

                        val sentence = sentenceBuilder.joinToString(" ")
                        val words = sentence.split(" ")

                        val sharedPreferences =
                            context.getSharedPreferences("MyAppPrefs", MODE_PRIVATE)
                        val language = sharedPreferences.getString("Language", "English")

//                        val prompt = "Give me the meaningful word or sentence without any explanation just give the word or sentence, from the following aphabets, ignore if a alphabet is repeated less than 3 times in consecutive series and consider which have more than 3. input: "+sentence

                        val prompt = "role: You are a professional sign language recogniser and sentence generator," +
                                "Task: You need to generate a meaningful sentence based on the given input data," +
                                "Assistant:  without any explanation give the exact output, avoid duplicate words from the data given, output must be a some meaningful word and not sentence from the given data" +
                                "language: " + language +
                                "Data: "+ sentence
                        Log.d("sentenc", sentence)
                        val response = processGeminiResponse(prompt)
                        sentenceBuilder.clear()
                        Log.d("GeminiResponse", response)
                        previous_response = response
                        flag = false
//                        return Pair(response, 0L)
                        return response
                    } else {
//                    Log.d("GeminiResponse", "Skipping Gemini API call to avoid flooding")
//                        return Pair(previous_response, 0L)
                        return previous_response
                    }
                }

                flag = true

                // Step 2: Get bounding box around the hand
                val handLandmarks = result.landmarks()[0]
                val boundingBox = getBoundingBox(handLandmarks, bitmap.width, bitmap.height)

                // Step 3: Crop the hand region
                val croppedBitmap = Bitmap.createBitmap(
                    bitmap,
                    boundingBox.left,
                    boundingBox.top,
                    boundingBox.width(),
                    boundingBox.height()
                )

                // Step 4: Preprocess the cropped hand image
                Log.d("alphabets model croppedBitmap", croppedBitmap.toString())
//            val floatBuffer = preprocessFrame(croppedBitmap)
                val landmarkArray = extractLandmarksFromImage(bitmap, result)

                Log.d("landmarkarray", landmarkArray?.size.toString())


                val floatBuffer = landmarkArray?.let { extractAdvancedFeatures(it) }


//            val floatBuffer = FloatArray(2079,)
                // Step 5: Create input tensor with correct shape
//            val shape = longArrayOf(1, 3, 224, 224)
//            val shape = longArrayOf(1, 2079)
//            val inputTensor = OnnxTensor.createTensor(env, floatBuffer, shape)

                val shape =
                    floatBuffer?.size?.let { longArrayOf(1, it.toLong()) } // e.g., [1, 2079]
                Log.d("nithin", floatBuffer?.size.toString())

                val inputTensor = floatBuffer?.let {
                    shape?.let { it1 ->
                        createInputTensor(
                            env, it,
                            it1
                        )
                    }
                }

                // Step 6: Run inference
                val resultMap =
                    session.run(mapOf(session.inputNames.iterator().next() to inputTensor))
                Log.d("resultmap", resultMap[0].value.toString())
                Log.d("resultmap_type", resultMap[0].value!!::class.java.name)

                val output2 = resultMap[0].value as LongArray
                Log.d("resultmap_values", output2.joinToString())


//            val outputTensor = resultMap[0].value as Array<FloatArray>
//            val outputArray = outputTensor[0]

                // Step 7: Postprocess the output
                val output_result = postprocessOutput(output2)

                sentenceBuilder.add(output_result.first)
//                return output_result
                return output_result.first
            }

        } catch (e: Exception) {
            Log.e("GestureProcessor", "Error recognizing gesture: ${e.message}", e)
            return "error"
        }
    }
    fun createInputTensor(env: OrtEnvironment, features: FloatArray, shape: LongArray): OnnxTensor {
        val floatBuffer = FloatBuffer.wrap(features)
        return OnnxTensor.createTensor(env, floatBuffer, shape)
    }

    fun extractLandmarksFromImage(
        bitmap: Bitmap,
        result: HandLandmarkerResult
    ): FloatArray? {
        val handLandmarksList = result.landmarks()

        if (handLandmarksList.isNotEmpty()) {
            val handLandmarks = handLandmarksList[0] // First detected hand
            val landmarks = FloatArray(21 * 3) // 21 landmarks with (x, y, z)

            for ((i, landmark) in handLandmarks.withIndex()) {
                landmarks[i * 3] = landmark.x()
                landmarks[i * 3 + 1] = landmark.y()
                landmarks[i * 3 + 2] = landmark.z()
            }

            return landmarks
        }

        return null
    }




    // Declare this at class level so it persists
    private var lastGeminiRequestTime = 0L

    // Utility to get bounding box from landmarks
//    import com.google.mediapipe.tasks.components.containers.NormalizedLandmark

    private fun getBoundingBox(landmarks: List<NormalizedLandmark>, imageWidth: Int, imageHeight: Int): Rect {
        var minX = 1f
        var minY = 1f
        var maxX = 0f
        var maxY = 0f

        for (landmark in landmarks) {
            minX = min(minX, landmark.x())
            minY = min(minY, landmark.y())
            maxX = max(maxX, landmark.x())
            maxY = max(maxY, landmark.y())
        }

        val left = (minX * imageWidth).toInt().coerceAtLeast(0)
        val top = (minY * imageHeight).toInt().coerceAtLeast(0)
        val right = (maxX * imageWidth).toInt().coerceAtMost(imageWidth - 1)
        val bottom = (maxY * imageHeight).toInt().coerceAtMost(imageHeight - 1)

        return Rect(left, top, right, bottom)
    }


}
