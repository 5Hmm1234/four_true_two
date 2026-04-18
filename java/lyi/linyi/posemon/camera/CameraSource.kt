package lyi.linyi.posemon.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.Camera as SystemCamera
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.WindowManager
import lyi.linyi.posemon.data.Camera as CameraEnum
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class CameraSource(
    private val surfaceView: SurfaceView,
    private val selectedCamera: CameraEnum,
    private val listener: CameraSourceListener?
) : SurfaceHolder.Callback, SystemCamera.PreviewCallback {

    private val TAG = "CameraSource"
    private var camera: SystemCamera? = null
    private var latestFrame: Bitmap? = null
    private val fpsExecutor = Executors.newSingleThreadScheduledExecutor()
    private var frameCount = 0
    private var lastFpsTime = System.currentTimeMillis()
    private val holder: SurfaceHolder = surfaceView.holder

    private val isProcessing = AtomicBoolean(false)
    // 提高检测频率，改为 30ms（最高允许30帧左右，提高响应速度）
    private val FRAME_INTERVAL_MS = 30L
    private var lastProcessTime = 0L

    private var displayRotationDegrees = 0

    init {
        holder.addCallback(this)
        holder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS)
    }

    fun getLatestFrame(): Bitmap? {
        return latestFrame
    }

    private fun initCamera() {
        try {
            camera = if (selectedCamera == CameraEnum.BACK) {
                SystemCamera.open(SystemCamera.CameraInfo.CAMERA_FACING_BACK)
            } else {
                SystemCamera.open(SystemCamera.CameraInfo.CAMERA_FACING_FRONT)
            }

            val parameters = camera!!.parameters
            parameters.previewFormat = ImageFormat.NV21

            if (parameters.supportedFocusModes.contains(SystemCamera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO)) {
                parameters.focusMode = SystemCamera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO
            }

            val size = getOptimalPreviewSize(parameters.supportedPreviewSizes, surfaceView.width, surfaceView.height)
            parameters.setPreviewSize(size.width, size.height)

            setCameraDisplayOrientation()
            camera!!.parameters = parameters
            camera!!.setPreviewCallback(this)
            camera!!.setPreviewDisplay(holder)
            startFpsCalculate()
        } catch (e: Exception) {
            Log.e(TAG, "相机初始化失败: ${e.message}", e)
        }
    }

    private fun setCameraDisplayOrientation() {
        val cameraInfo = SystemCamera.CameraInfo()
        SystemCamera.getCameraInfo(if (selectedCamera == CameraEnum.BACK) 0 else 1, cameraInfo)
        val wm = surfaceView.context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
        val rotation = wm.defaultDisplay.rotation
        var degrees = when (rotation) {
            android.view.Surface.ROTATION_0 -> 0
            android.view.Surface.ROTATION_90 -> 90
            android.view.Surface.ROTATION_180 -> 180
            else -> 270
        }
        val result = if (cameraInfo.facing == SystemCamera.CameraInfo.CAMERA_FACING_FRONT) {
            (360 - (cameraInfo.orientation + degrees) % 360) % 360
        } else {
            (cameraInfo.orientation - degrees + 360) % 360
        }

        displayRotationDegrees = result
        camera!!.setDisplayOrientation(result)
    }

    fun start() { camera?.startPreview() }
    fun stop() { try { camera?.stopPreview() } catch (_: Exception) {} }

    fun close() {
        try {
            camera?.setPreviewCallback(null)
            camera?.stopPreview()
            camera?.release()
            latestFrame?.recycle()
            fpsExecutor.shutdown()
            camera = null
            latestFrame = null
        } catch (_: Exception) {}
    }

    override fun onPreviewFrame(data: ByteArray?, camera: SystemCamera?) {
        if (data == null || camera == null) return
        frameCount++

        val currentTime = System.currentTimeMillis()
        if (currentTime - lastProcessTime < FRAME_INTERVAL_MS) return
        if (!isProcessing.compareAndSet(false, true)) return

        lastProcessTime = currentTime

        Thread {
            try {
                val size = camera.parameters.previewSize
                val yuvImage = YuvImage(data, ImageFormat.NV21, size.width, size.height, null)
                val stream = ByteArrayOutputStream()
                yuvImage.compressToJpeg(Rect(0, 0, size.width, size.height), 80, stream)
                val rawBitmap = BitmapFactory.decodeByteArray(stream.toByteArray(), 0, stream.size())

                val matrix = Matrix()
                matrix.postRotate(displayRotationDegrees.toFloat())
                val rotatedBitmap = Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)

                if (rotatedBitmap != rawBitmap) {
                    rawBitmap.recycle()
                }

                latestFrame?.recycle()
                latestFrame = rotatedBitmap

                listener?.onDetectedInfo(null, null)
            } catch (e: Exception) {
            } finally {
                isProcessing.set(false)
            }
        }.start()
    }

    private fun startFpsCalculate() {
        fpsExecutor.scheduleAtFixedRate({
            val now = System.currentTimeMillis()
            val elapsed = now - lastFpsTime
            if (elapsed > 0) {
                val fps = (frameCount * 1000 / elapsed).toInt()
                listener?.onFPSListener(fps)
            }
            frameCount = 0
            lastFpsTime = now
        }, 0, 1000, java.util.concurrent.TimeUnit.MILLISECONDS)
    }

    private fun getOptimalPreviewSize(sizes: List<SystemCamera.Size>, w: Int, h: Int): SystemCamera.Size {
        val targetRatio = w.toDouble() / h
        var optimalSize = sizes[0]
        var minDiff = Double.MAX_VALUE

        for (size in sizes) {
            val ratio = size.width.toDouble() / size.height
            if (Math.abs(ratio - targetRatio) < 0.1) {
                val diff = Math.abs(size.height - h).toDouble()
                if (diff < minDiff) {
                    optimalSize = size
                    minDiff = diff
                }
            }
        }
        return optimalSize
    }

    override fun surfaceCreated(holder: SurfaceHolder) { initCamera() }
    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        stop(); start()
    }
    override fun surfaceDestroyed(holder: SurfaceHolder) { close() }

    interface CameraSourceListener {
        fun onFPSListener(fps: Int)
        fun onDetectedInfo(personScore: Float?, poseLabels: List<Pair<String, Float>>?)
    }
}