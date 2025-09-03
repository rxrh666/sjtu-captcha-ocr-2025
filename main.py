import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import base64
import traceback
from flask import jsonify

# 配置參數，與 JavaScript 中的設定保持一致
WIDTH = 110
HEIGHT = 40
THRESHOLD = 157

def cvt_binary(data, threshold):
    """
    將圖像數據轉換為二值化數據，對應 JavaScript 中的 cvtBinary 函數。
    """
    bin_pic = []
    for i in range(0, len(data), 4):
        # 提取 RGB 通道值
        r = data[i]
        g = data[i+1]
        b = data[i+2]
        # 計算亮度
        luma = r * 299 / 1000 + g * 587 / 1000 + b * 114 / 1000
        # 二值化處理
        bin_pic.append(0.0 if luma < threshold else 1.0)
    return np.array(bin_pic, dtype=np.float32)

def predict_captcha(image_data):
    """
    使用 ONNX 模型進行驗證碼預測。
    該函數設計為單例模式，以避免重複加載模型。
    """
    # 加載模型，只在首次呼叫時執行
    if not hasattr(predict_captcha, 'session'):
        predict_captcha.session = ort.InferenceSession(
            "model.onnx", 
            providers=["CPUExecutionProvider"]
        )
    
    # 從記憶體讀取圖像數據
    with Image.open(io.BytesIO(image_data)) as img:
        # 調整圖像尺寸並轉換為 RGBA 格式
        img = img.resize((WIDTH, HEIGHT))
        img = img.convert("RGBA")
        raw_data = list(img.getdata())
        raw_img = []
        for pixel in raw_data:
            raw_img.extend(pixel)
    
    # 預處理：將圖像二值化
    processed_img = cvt_binary(raw_img, THRESHOLD)
    # 準備輸入張量，將其重塑為模型所需的形狀
    input_tensor = processed_img.reshape(1, 1, HEIGHT, WIDTH)
    
    # 執行推理
    output_map = predict_captcha.session.run(None, {"input.1": input_tensor})
    
    # 解析輸出結果
    result = ""
    # 遍歷所有輸出結果
    for output in output_map:
        # 找到概率最大的索引
        max_index = np.argmax(output)
        # 忽略索引 26 (可能代表空白或無效字符)
        if max_index != 26:
            # 將索引轉換為對應的字母
            result += chr(ord('a') + max_index)
    
    return result

# --- Cloud Functions 的入口函數 ---
def ocr_captcha(request):
    """
    Cloud Functions 的入口點。
    `request` 對象是 Flask 的請求對象。
    """
    try:
        # 獲取 JSON 請求體
        request_data = request.get_json(silent=True)
        if not request_data or 'imageData' not in request_data:
            return jsonify({'error': 'No JSON data or imageData field provided'}), 400

        # 從 JSON 數據中獲取 Base64 字串並解碼
        encoded_image_string = request_data['imageData']
        image_data = base64.b64decode(encoded_image_string)

        captcha_text = predict_captcha(image_data)
        
        # 以 JSON 格式返回識別結果
        return jsonify({'captcha': captcha_text})

    except Exception as e:
        # 捕獲所有異常並返回錯誤訊息
        print("An error occurred:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
