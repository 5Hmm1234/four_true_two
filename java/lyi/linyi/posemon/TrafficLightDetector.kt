package lyi.linyi.posemon

import android.content.Context
import android.graphics.Bitmap
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream

class TrafficLightDetector private constructor() {

    private lateinit var module: Module

    companion object {
        @Volatile
        private var INSTANCE: TrafficLightDetector? = null

        fun getInstance() = INSTANCE ?: synchronized(this) {
            TrafficLightDetector().also { INSTANCE = it }
        }

        fun initModel(context: Context) {
            getInstance().initModelInternal(context)
        }
    }

    private fun initModelInternal(context: Context) {
        if (!::module.isInitialized) {
            try {
                val path = assetFilePath(context, "trafficlight.pt")
                module = Module.load(path)
            } catch (e: Exception) {
            }
        }
    }

    fun detect(bitmap: Bitmap): String {
        if (!::module.isInitialized) return "none"
        return try {
            val resized = Bitmap.createScaledBitmap(bitmap, 320, 320, true)
            val tensor = TensorImageUtils.bitmapToFloat32Tensor(
                resized,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
            )
            val output = module.forward(IValue.from(tensor)).toTensor()
            val data = output.dataAsFloatArray

            var result = "none"
            for (i in data.indices step 6) {
                val conf = data[i + 4]
                val cls = data[i + 5].toInt()
                if (conf > 0.2f) {
                    // ✅ 修复语法错误：else 后添加 ->
                    result = when (cls) {
                        0 -> "red"
                        1 -> "green"
                        2 -> "yellow"
                        else -> "none"
                    }
                    break
                }
            }
            resized.recycle()
            result
        } catch (e: Exception) {
            "none"
        }
    }

    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists()) return file.absolutePath
        try {
            context.assets.open(assetName).use { input ->
                FileOutputStream(file).use { output ->
                    input.copyTo(output)
                }
            }
        } catch (e: Exception) {}
        return file.absolutePath
    }
}