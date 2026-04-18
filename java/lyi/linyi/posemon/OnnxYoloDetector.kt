package lyi.linyi.posemon

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import ai.onnxruntime.*
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

class OnnxYoloDetector(context: Context) {
    // ⚠️ 盲人关乎生命安全，如果实测红绿反了，请互换 "go" 和 "stop" 的位置
    private val classes = arrayOf(
        "blank", "countdown_blank", "countdown_go",
        "countdown_stop", "crossing", "go", "stop"
    )

    private val TAG = "OnnxYoloDetector"
    private val IOU_THRESHOLD = 0.45f
    private val INPUT_SIZE = 640

    private lateinit var ortEnv: OrtEnvironment
    private lateinit var ortSession: OrtSession
    private var isModelLoaded = false

    init {
        try {
            copyModelFromAssets(context)
            ortEnv = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions()
            options.setIntraOpNumThreads(4)
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

            // 禁用NNAPI硬件加速，因为当前模型不支持
            Log.d(TAG, "禁用NNAPI硬件加速，使用CPU运行模型")
            // 尝试使用CPU运行模型，避免NNAPI兼容性问题

            val modelPath = File(context.filesDir, "trafficlight.onnx").absolutePath
            Log.d(TAG, "模型路径: $modelPath")
            Log.d(TAG, "模型文件存在: ${File(modelPath).exists()}")
            
            ortSession = ortEnv.createSession(modelPath, options)
            isModelLoaded = true
            Log.d(TAG, "模型加载成功！输入: ${ortSession.inputNames}")
        } catch (e: Exception) {
            Log.e(TAG, "模型加载失败: ${e.message}", e)
            isModelLoaded = false
        }
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        if (!isModelLoaded || !::ortSession.isInitialized) {
            Log.w(TAG, "模型未加载，跳过检测")
            return emptyList()
        }

        return try {
            Log.d(TAG, "开始检测， bitmap size: ${bitmap.width}x${bitmap.height}")
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
            val inputArray = bitmapToFloatArray(resizedBitmap)
            resizedBitmap.recycle()

            val inputName = ortSession.inputNames.iterator().next()
            val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
            val inputTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(inputArray), shape)

            val outputMap = ortSession.run(mapOf(inputName to inputTensor))
            val outputTensor = outputMap.iterator().next().value as OnnxTensor

            val outputBuffer = outputTensor.floatBuffer
            val outputArray = FloatArray(outputBuffer.remaining())
            outputBuffer.get(outputArray)

            Log.d(TAG, "模型输出数组大小: ${outputArray.size}")

            // 解析 YOLOv8 输出
            val results = parseOutput(outputArray, bitmap.width, bitmap.height)
            Log.d(TAG, "检测到 ${results.size} 个目标: ${results.map { "${it.label}(${it.confidence})" }}")

            inputTensor.close()
            outputTensor.close()
            outputMap.close()

            results
        } catch (e: Exception) {
            Log.e(TAG, "检测过程出错: ${e.message}", e)
            emptyList()
        }
    }

    private fun copyModelFromAssets(context: Context) {
        val modelFile = File(context.filesDir, "trafficlight.onnx")
        Log.d(TAG, "模型文件路径: ${modelFile.absolutePath}")
        Log.d(TAG, "模型文件存在: ${modelFile.exists()}")
        if (modelFile.exists()) {
            Log.d(TAG, "模型文件已存在，跳过复制")
            return
        }
        try {
            Log.d(TAG, "开始从assets复制模型文件")
            context.assets.open("trafficlight.onnx").use { input ->
                FileOutputStream(modelFile).use { output -> input.copyTo(output) }
            }
            Log.d(TAG, "模型文件复制成功")
        } catch (e: Exception) {
            Log.e(TAG, "模型文件复制失败: ${e.message}", e)
        }
    }

    private fun bitmapToFloatArray(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        val floatArray = FloatArray(3 * INPUT_SIZE * INPUT_SIZE)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            floatArray[i] = ((pixel shr 16) and 0xFF) / 255.0f
            floatArray[i + INPUT_SIZE * INPUT_SIZE] = ((pixel shr 8) and 0xFF) / 255.0f
            floatArray[i + 2 * INPUT_SIZE * INPUT_SIZE] = (pixel and 0xFF) / 255.0f
        }
        return floatArray
    }

    private fun parseOutput(output: FloatArray, imgW: Int, imgH: Int): List<Detection> {
        val detections = mutableListOf<Detection>()
        val numClasses = classes.size
        val numBoxes = output.size / (numClasses + 4)

        Log.d(TAG, "解析输出: numBoxes=$numBoxes, numClasses=$numClasses, outputSize=${output.size}")

        for (i in 0 until numBoxes) {
            val base = i * (numClasses + 4)
            if (base + numClasses + 4 > output.size) break

            val x = output[base]
            val y = output[base + 1]
            val w = output[base + 2]
            val h = output[base + 3]

            var maxConf = 0f
            var cls = 0
            for (j in 0 until numClasses) {
                val conf = output[base + 4 + j]
                if (conf > maxConf) {
                    maxConf = conf
                    cls = j
                }
            }

            val isValid = when (cls) {
                5 -> maxConf >= 0.2f  // 降低绿灯阈值，提高灵敏度
                6 -> maxConf >= 0.25f // 保持红灯阈值不变
                else -> false
            }
            if (!isValid) continue

            val rect = RectF(
                (x - w/2) * imgW / INPUT_SIZE,
                (y - h/2) * imgH / INPUT_SIZE,
                (x + w/2) * imgW / INPUT_SIZE,
                (y + h/2) * imgH / INPUT_SIZE
            )

            val lightSize = rect.width() * rect.height()
            val minSize = imgW * imgH * 0.0005f
            val maxSize = imgW * imgH * 0.15f
            val isLight = lightSize in minSize..maxSize

            if (isLight) {
                detections.add(Detection(rect, classes[cls], maxConf))
                Log.d(TAG, "有效检测: ${classes[cls]}, conf=$maxConf, size=$lightSize")
            }
        }
        return nms(detections)
    }

    private fun nms(detections: List<Detection>): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }
        val keep = mutableListOf<Detection>()
        for (d in sorted) {
            var keepFlag = true
            for (k in keep) {
                if (calculateIoU(d.rect, k.rect) > IOU_THRESHOLD) {
                    keepFlag = false
                    break
                }
            }
            if (keepFlag) keep.add(d)
        }
        return keep
    }

    private fun calculateIoU(a: RectF, b: RectF): Float {
        val l = max(a.left, b.left)
        val t = max(a.top, b.top)
        val r = min(a.right, b.right)
        val btm = min(a.bottom, b.bottom)
        val w = max(0f, r - l)
        val h = max(0f, btm - t)
        val intersection = w * h
        val areaA = (a.right - a.left) * (a.bottom - a.top)
        val areaB = (b.right - b.left) * (b.bottom - b.top)
        return intersection / (areaA + areaB - intersection)
    }

    data class Detection(val rect: RectF, val label: String, val confidence: Float)

    fun release() {
        try { ortSession.close(); ortEnv.close() } catch (_: Exception) {}
    }
}