package lyi.linyi.posemon

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.view.KeyEvent
import android.view.SurfaceView
import android.view.WindowManager
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import lyi.linyi.posemon.camera.CameraSource
import lyi.linyi.posemon.data.Camera
import java.util.LinkedList
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var tts: TextToSpeech
    private lateinit var tvTopHint: TextView
    private lateinit var surfaceView: SurfaceView
    private lateinit var tvFPS: TextView

    private lateinit var detector: OnnxYoloDetector
    private var cameraSource: CameraSource? = null
    private val selectedCamera = Camera.BACK

    // 🚀 加快响应速度：缓存池从5帧降为3帧
    private val HISTORY_SIZE = 3 // 减少历史窗口，提高响应速度3
    private val stateHistory = LinkedList<String>()
    private var finalLightState = "none"
    private var lastAnnounceTime = 0L

    // 缩短重复播报间隔，让盲人更有安全感
    private val RED_ANNOUNCE_INTERVAL = 3000L   // 红灯每3秒提醒一次
    private val GREEN_ANNOUNCE_INTERVAL = 2000L // 绿灯每2秒提醒一次

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        surfaceView = findViewById(R.id.surfaceView)
        tvTopHint = findViewById(R.id.tvTopHint)
        tvFPS = findViewById(R.id.tvFps)

        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts.language = Locale.CHINA
                tts.setSpeechRate(1.3f) // 语速稍快，紧急情况更敏捷
            }
        }

        println("开始初始化检测器")
        detector = OnnxYoloDetector(this)
        println("检测器初始化完成")
        tvTopHint.text = "盲人导航已启动，请将摄像头对准前方"

        if (!checkPermission()) requestPermission()
    }

    private fun openCamera() {
        if (cameraSource == null) {
            println("开始打开相机")
            cameraSource = CameraSource(surfaceView, selectedCamera, object : CameraSource.CameraSourceListener {
                override fun onFPSListener(fps: Int) {
                    runOnUiThread { tvFPS.text = "FPS:$fps" }
                }
                override fun onDetectedInfo(personScore: Float?, poseLabels: List<Pair<String, Float>>?) {
                    val bitmap = cameraSource?.getLatestFrame()
                    println("获取到帧: $bitmap")
                    if (bitmap != null && !bitmap.isRecycled) {
                        println("开始处理帧")
                        processDetection(bitmap)
                        println("处理帧完成")
                    } else {
                        println("帧为空或已回收")
                    }
                }
            })
            println("相机打开成功")
        }
    }

    private fun processDetection(bitmap: Bitmap) {
        try {
            println("开始检测")
            val results = detector.detect(bitmap)
            println("检测结果: $results")

            val redDetection = results.find { it.label == "stop" && it.confidence > 0.25f }
            val greenDetection = results.find { it.label == "go" && it.confidence > 0.25f }

            val newResult = when {
                redDetection != null && greenDetection != null -> {
                    if (redDetection.confidence > greenDetection.confidence) "stop" else "go"
                }
                redDetection != null -> "stop"
                greenDetection != null -> "go"
                else -> "none"
            }

            println("处理结果: $newResult")
            updateStateMachine(newResult)
        } catch (e: Exception) {
            println("检测过程出错: ${e.message}")
            e.printStackTrace()
        }
    }

    private fun updateStateMachine(currentFrameResult: String) {
        stateHistory.add(currentFrameResult)
        if (stateHistory.size > HISTORY_SIZE) {
            stateHistory.removeFirst()
        }

        // 直接使用最新状态，提高响应速度
        val validatedState = currentFrameResult

        if (validatedState != finalLightState) {
            finalLightState = validatedState
            // 状态一旦发生改变，强制立刻清空TTS队列播报
            checkStateChange(isStateChanged = true)
        } else {
            checkStateChange(isStateChanged = false)
        }
    }

    private fun checkStateChange(isStateChanged: Boolean) {
        runOnUiThread {
            val currentTime = System.currentTimeMillis()
            val timeSinceLastAnnounce = currentTime - lastAnnounceTime

            when (finalLightState) {
                "stop" -> {
                    tvTopHint.text = "红灯 禁止通行"
                    tvTopHint.setBackgroundColor(Color.RED)
                    if (isStateChanged || timeSinceLastAnnounce > RED_ANNOUNCE_INTERVAL) {
                        // QUEUE_FLUSH 会打断正在说的话，用于紧急变灯提醒
                        tts.speak("红灯，禁止通行", TextToSpeech.QUEUE_FLUSH, null, null)
                        lastAnnounceTime = currentTime
                    }
                }
                "go" -> {
                    tvTopHint.text = "绿灯 可以通行"
                    tvTopHint.setBackgroundColor(Color.GREEN)
                    if (isStateChanged || timeSinceLastAnnounce > GREEN_ANNOUNCE_INTERVAL) {
                        tts.speak("绿灯，请通行", TextToSpeech.QUEUE_FLUSH, null, null)
                        lastAnnounceTime = currentTime
                    }
                }
                else -> {
                    tvTopHint.text = "正在寻找红绿灯..."
                    tvTopHint.setBackgroundColor(Color.GRAY)
                }
            }
        }
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        if (keyCode == KeyEvent.KEYCODE_VOLUME_UP || keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) {
            announceCurrentState()
            return true
        }
        return super.onKeyDown(keyCode, event)
    }

    private fun announceCurrentState() {
        when (finalLightState) {
            "stop" -> tts.speak("前方红灯，不要走", TextToSpeech.QUEUE_FLUSH, null, null)
            "go" -> tts.speak("前方绿灯，可以过马路", TextToSpeech.QUEUE_FLUSH, null, null)
            else -> tts.speak("未发现红绿灯", TextToSpeech.QUEUE_FLUSH, null, null)
        }
    }

    private fun checkPermission() = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    private fun requestPermission() = ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1001)

    override fun onStart() {
        super.onStart()
        println("onStart: 检查权限")
        if (checkPermission()) {
            println("onStart: 权限已授予，打开相机")
            openCamera()
            cameraSource?.start()
            println("onStart: 相机已启动")
        } else {
            println("onStart: 权限未授予")
        }
    }
    


    override fun onPause() {
        super.onPause()
        cameraSource?.stop()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraSource?.close()
        tts.shutdown()
        detector.release()
    }

    override fun onRequestPermissionsResult(code: Int, p: Array<out String>, g: IntArray) {
        super.onRequestPermissionsResult(code, p, g)
        if (g.isNotEmpty() && g[0] == PackageManager.PERMISSION_GRANTED) {
            println("权限已授予，重新创建活动")
            recreate()
        }
    }
}