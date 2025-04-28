package com.example.slconnxmodel

import android.util.Log
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException

object GeminiApiHelper {
    private const val API_KEY = "AIzaSyBJUIbnEIWvGmcUDl4BP7uhbltSXGt86Uo"
    private const val BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    private val client = OkHttpClient()

    fun sendPrompt(prompt: String, callback: (String?) -> Unit) {
        val requestBody = JSONObject().apply {
            put("contents", JSONArray().apply {
                put(JSONObject().apply {
                    put("parts", JSONArray().apply {
                        put(JSONObject().apply {
                            put("text", prompt)
                        })
                    })
                })
            })
        }.toString()

        val request = Request.Builder()
            .url("$BASE_URL?key=$API_KEY")
            .post(RequestBody.create("application/json; charset=utf-8".toMediaTypeOrNull(), requestBody))
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e("GeminiApiHelper", "Request failed: ${e.message}", e)
                callback(null)
            }

            override fun onResponse(call: Call, response: Response) {
                response.use {
                    if (!response.isSuccessful) {
                        Log.e("GeminiApiHelper", "Unexpected code $response")
                        callback(null)
                    } else {
                        val responseString = response.body?.string()
                        val result = parseResponse(responseString)
                        callback(result)
                    }
                }
            }
        })
    }

    private fun parseResponse(responseString: String?): String? {
        return try {
            val jsonObject = JSONObject(responseString ?: "")
            val candidates = jsonObject.getJSONArray("candidates")
            val firstCandidate = candidates.getJSONObject(0)
            val content = firstCandidate.getJSONObject("content")
            val parts = content.getJSONArray("parts")
            parts.getJSONObject(0).getString("text")
        } catch (e: Exception) {
            Log.e("GeminiApiHelper", "Error parsing response: ${e.message}", e)
            null
        }
    }
}
