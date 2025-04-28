package com.example.slconnxmodel

import android.graphics.Bitmap
import android.graphics.BitmapFactory

object BitmapUtils {
    fun bytesToBitmap(bytes: ByteArray, width: Int, height: Int): Bitmap {
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }
}
