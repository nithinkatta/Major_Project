package com.example.slconnxmodel
import ai.onnxruntime.*
import android.content.Context
import android.graphics.Bitmap

class OnnxGestureProcessor(context: Context) {

    private val ortEnv = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val modelBytes = context.assets.open("model.onnx").readBytes()
        session = ortEnv.createSession(modelBytes)
    }

    fun classify(bitmap: Bitmap): String {
        val inputTensor = preprocessBitmap(bitmap)

        val output = session.run(mapOf(session.inputNames.iterator().next() to inputTensor))
        val probabilities = output[0].value as FloatArray

        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: -1
        return labelFromIndex(maxIndex)
    }

    private fun preprocessBitmap(bitmap: Bitmap): OnnxTensor {
        // Resize, normalize, etc. based on your model input
        TODO("Implement preprocessing")
    }

    private fun labelFromIndex(index: Int): String {
        val labels = listOf("Open Palm", "Fist", "Thumbs Up", "Peace", "Ok")
        return labels.getOrElse(index) { "Unknown" }
    }
}
