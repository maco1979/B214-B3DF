// 获取DOM元素
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const photoPreview = document.getElementById('photo-preview');
const statusElement = document.getElementById('status');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const photoBtn = document.getElementById('photo-btn');
const switchBtn = document.getElementById('switch-btn');
const clearBtn = document.getElementById('clear-btn');

// 全局变量
let stream = null;
let currentCameraIndex = 0;
let availableCameras = [];

// 初始化
function init() {
    // 检查浏览器是否支持摄像头API
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showStatus('您的浏览器不支持摄像头API', 'error');
        startBtn.disabled = true;
        return;
    }
    
    // 事件监听器
    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click', stopCamera);
    photoBtn.addEventListener('click', takePhoto);
    switchBtn.addEventListener('click', switchCamera);
    clearBtn.addEventListener('click', clearPhoto);
    
    // 获取可用摄像头
    getAvailableCameras();
}

// 获取可用摄像头列表
async function getAvailableCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        availableCameras = devices.filter(device => device.kind === 'videoinput');
        
        if (availableCameras.length === 0) {
            showStatus('未检测到可用摄像头', 'error');
            startBtn.disabled = true;
        } else {
            showStatus(`检测到 ${availableCameras.length} 个可用摄像头`, 'success');
            if (availableCameras.length > 1) {
                switchBtn.disabled = false;
            }
        }
    } catch (error) {
        showStatus(`获取摄像头列表失败: ${error.message}`, 'error');
    }
}

// 开始摄像头
async function startCamera() {
    try {
        showStatus('正在启动摄像头...', 'info');
        
        // 获取摄像头流
        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        };
        
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        
        // 将流分配给视频元素
        video.srcObject = stream;
        
        // 更新按钮状态
        startBtn.disabled = true;
        stopBtn.disabled = false;
        photoBtn.disabled = false;
        
        showStatus('摄像头已启动', 'success');
        
    } catch (error) {
        let errorMessage = '启动摄像头失败';
        if (error.name === 'NotAllowedError') {
            errorMessage = '未获得摄像头使用权限，请在浏览器设置中允许';
        } else if (error.name === 'NotFoundError') {
            errorMessage = '未检测到摄像头设备';
        } else {
            errorMessage = `启动摄像头失败: ${error.message}`;
        }
        showStatus(errorMessage, 'error');
    }
}

// 停止摄像头
function stopCamera() {
    if (stream) {
        // 停止所有轨道
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        
        // 更新视频元素
        video.srcObject = null;
        
        // 更新按钮状态
        startBtn.disabled = false;
        stopBtn.disabled = true;
        photoBtn.disabled = true;
        
        showStatus('摄像头已停止', 'info');
    }
}

// 拍照
function takePhoto() {
    if (!stream) return;
    
    try {
        // 设置canvas大小与视频相同
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // 绘制当前视频帧到canvas
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // 将canvas内容转换为图片
        const photoDataUrl = canvas.toDataURL('image/jpeg');
        
        // 显示照片预览
        photoPreview.src = photoDataUrl;
        photoPreview.style.display = 'block';
        
        // 更新按钮状态
        clearBtn.disabled = false;
        
        showStatus('拍照成功', 'success');
        
    } catch (error) {
        showStatus(`拍照失败: ${error.message}`, 'error');
    }
}

// 切换摄像头
async function switchCamera() {
    if (availableCameras.length <= 1) return;
    
    // 计算下一个摄像头索引
    currentCameraIndex = (currentCameraIndex + 1) % availableCameras.length;
    
    // 停止当前摄像头
    stopCamera();
    
    // 启动新摄像头
    await startCamera();
}

// 清除照片
function clearPhoto() {
    photoPreview.style.display = 'none';
    photoPreview.src = '';
    clearBtn.disabled = true;
    showStatus('照片已清除', 'info');
}

// 显示状态信息
function showStatus(message, type = 'info') {
    statusElement.textContent = message;
    
    // 根据类型设置不同的背景颜色
    switch(type) {
        case 'success':
            statusElement.style.backgroundColor = '#d4edda';
            statusElement.style.color = '#155724';
            break;
        case 'error':
            statusElement.style.backgroundColor = '#f8d7da';
            statusElement.style.color = '#721c24';
            break;
        case 'warning':
            statusElement.style.backgroundColor = '#fff3cd';
            statusElement.style.color = '#856404';
            break;
        default:
            statusElement.style.backgroundColor = '#d1ecf1';
            statusElement.style.color = '#0c5460';
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', init);