import json
import os
import random
import re
import string
import time
import warnings
import math
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import requests
import execjs
from loguru import logger
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import queue
import threading
from functools import lru_cache
from fake_useragent import UserAgent

warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)
warnings.filterwarnings("ignore", message=".*SIFT_create.*deprecated.*")

DEBUG = False

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
USE_CUDA = True if torch.cuda.is_available() else False
DEVICE = 'cuda' if USE_CUDA else 'cpu'

# ============ OPTIMIZED CONSTANTS ============
if USE_CUDA:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# ============ EMERGENCY ERROR HANDLER ============
def emergency_fallback():
    """Emergency fallback when all else fails"""
    return [(80, 70), (160, 120), (240, 90)]

def safe_list_access(lst, index, default=None):
    """Ultra-safe list access"""
    try:
        if lst is None or not isinstance(lst, (list, tuple)):
            return default
        if not (0 <= index < len(lst)):
            return default
        return lst[index]
    except:
        return default

# ============ MODEL STATE ============
_model_state = None
_model_lock = threading.Lock()

def initialize_global_model():
    global _model_state
    
    if _model_state is not None:
        return _model_state
        
    with _model_lock:
        if _model_state is not None:
            return _model_state
            
        model_path = os.path.join(DIR_PATH, 'net.pkl')
        if not os.path.exists(model_path):
            logger.error("Model file net.pkl not found")
            return None
            
        try:
            import pickle
            import io

            class _ModelUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if name == 'Mini':
                        return Mini
                    if name == 'ConvBN':
                        return Mini.ConvBN
                    return super().find_class(module, name)

            with open(model_path, 'rb') as f:
                state = torch.load(f, map_location=torch.device(DEVICE), weights_only=False, pickle_module=type('M', (), {
                    'Unpickler': _ModelUnpickler,
                    'load': pickle.load,
                    'UnpicklingError': pickle.UnpicklingError,
                }))
            
            if 'net' in state:
                state['net'] = state['net'].to(DEVICE)
                state['net'].eval()
                
                if USE_CUDA:
                    state['net'] = state['net'].half()
                
            _model_state = state
            logger.success(f"Model loaded on {DEVICE}")
            return _model_state
            
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            return None

def get_global_model():
    global _model_state
    if _model_state is None:
        return initialize_global_model()
    return _model_state

# ============ JS CACHE ============
@lru_cache(maxsize=5)
def get_compiled_js_cached(file_name):
    try:
        js_path = os.path.join(DIR_PATH, file_name)

        print(f"[YIDUN] Loading JS: {js_path}")

        with open(js_path, "r", encoding="utf-8") as f:
            js_code = f.read()

        print(f"[YIDUN] JS loaded: {len(js_code)} characters")

        ctx = execjs.compile(js_code)

        print("[YIDUN] JS compiled successfully")
        return ctx

    except Exception as e:
        print(f"[YIDUN] JS ERROR: {type(e).__name__}: {e}")
        return None


def get_compiled_js(file_name):
    return get_compiled_js_cached(file_name)

# ============ SIFT DETECTOR ============
_sift_detector = None
_sift_lock = threading.Lock()

def get_sift_detector():
    global _sift_detector
    if _sift_detector is None:
        with _sift_lock:
            if _sift_detector is None:
                try:
                    _sift_detector = cv2.SIFT_create(nfeatures=50, contrastThreshold=0.08)
                    logger.info("Using cv2.SIFT_create()")
                except AttributeError:
                    try:
                        _sift_detector = cv2.xfeatures2d.SIFT_create(nfeatures=50, contrastThreshold=0.08)
                        logger.info("Using cv2.xfeatures2d.SIFT_create()")
                    except AttributeError:
                        logger.warning("SIFT not available, falling back to ORB")
                        _sift_detector = cv2.ORB_create(nfeatures=50)
    return _sift_detector

# ============ CONSTANTS ============
file_lock = threading.Lock()
TOKEN_OUTPUT_FILE = os.path.join(DIR_PATH, 'validated_tokens.txt') # Added for local saving

# Configuration from bypasser-og.py
REFERER = "https://mtacc.mobilelegends.com/"
ID = "fef5c67c39074e9d845f4bf579cc07af"
FP_H = "mtacc.mobilelegends.com"

DUN163_DOMAINS = [
    "https://c.dun.163.com",
    "https://c.dun.163yun.com"
]

# ============ API CONFIG (REMOVED) ============
# API_SUBMIT_URL = "http://10.63.197.72:5000/api/submit"
# API_KEY = "BulletPH"
# ====================================

def rotate_about_center(src, angle, scale=1.):
    try:
        w = src.shape[1]
        h = src.shape[0]
        rangle = np.deg2rad(angle)
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
        rot_mat[0,2] += rot_move[0]
        rot_mat[1,2] += rot_move[1]
        return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LINEAR)
    except:
        return src

def parse_y_pred(ypred, anchors, class_types, islist=False, threshold=0.2, nms_threshold=0):
    """EMERGENCY SAFE VERSION - Will not crash on index errors"""
    try:
        if not anchors or not class_types:
            return [] if islist else None
            
        ceillen = 5 + len(class_types)
        sigmoid = lambda x: 1/(1+math.exp(-x))
        infos = []
        
        for idx in range(min(len(anchors), 3)):  # Limit to max 3 anchors
            try:
                tensor_idx = 4 + idx * ceillen
                if tensor_idx >= ypred.shape[3]:
                    continue
                    
                if USE_CUDA:
                    a = ypred[:,:,:,tensor_idx].cpu().detach().numpy()
                else:
                    a = ypred[:,:,:,tensor_idx].detach().numpy()
                    
                for ii, i in enumerate(a[0]):
                    for jj, j in enumerate(i):
                        infos.append((ii, jj, idx, sigmoid(j)))
            except:
                continue
                
        if not infos:
            return [] if islist else None
            
        infos = sorted(infos, key=lambda i: -i[3])
        
        def get_xyxy_clz_con_emergency(info):
            """Emergency version that will not crash"""
            try:
                gap = 416/ypred.shape[1]
                x, y, idx, con = info
                
                if idx >= len(anchors):
                    return None
                    
                gp = idx * ceillen
                
                # Check tensor bounds before access
                if (gp + 5 + len(class_types)) > ypred.shape[3]:
                    return None
                    
                contain = torch.sigmoid(ypred[0, x, y, gp+4])
                pred_xy = torch.sigmoid(ypred[0, x, y, gp+0:gp+2])
                pred_wh = ypred[0, x, y, gp+2:gp+4]
                pred_clz = ypred[0, x, y, gp+5:gp+5+len(class_types)]
                
                if USE_CUDA:
                    pred_xy = pred_xy.cpu().detach().numpy()
                    pred_wh = pred_wh.cpu().detach().numpy()
                    pred_clz = pred_clz.cpu().detach().numpy()
                else:
                    pred_xy = pred_xy.detach().numpy()
                    pred_wh = pred_wh.detach().numpy()
                    pred_clz = pred_clz.detach().numpy()
                    
                exp = math.exp
                cx, cy = float(pred_xy[0]), float(pred_xy[1])
                rx, ry = (cx + x)*gap, (cy + y)*gap
                rw, rh = float(pred_wh[0]), float(pred_wh[1])
                rw, rh = exp(rw)*anchors[idx][0], exp(rh)*anchors[idx][1]
                clz_ = [float(x) for x in pred_clz]
                xx = rx - rw/2
                _x = rx + rw/2
                yy = ry - rh/2
                _y = ry + rh/2
                
                if USE_CUDA:
                    log_cons = torch.sigmoid(ypred[:,:,:,gp+4]).cpu().detach().numpy()
                else:
                    log_cons = torch.sigmoid(ypred[:,:,:,gp+4]).detach().numpy()
                    
                log_cons = np.transpose(log_cons, (0, 2, 1))
                
                # Safe class lookup
                clz = 'unknown'
                if clz_:
                    max_val = max(clz_)
                    max_idx = clz_.index(max_val)
                    for key, value in class_types.items():
                        if value == max_idx:
                            clz = key
                            break
                            
                return [xx, yy, _x, _y], clz, con, log_cons
                
            except:
                return None
        
        if islist:
            limited_infos = infos[:min(50, len(infos))]
            v = []
            for i in limited_infos:
                if i[3] > threshold:
                    result = get_xyxy_clz_con_emergency(i)
                    if result is not None:
                        v.append(result)
            return v
        else:
            if infos:
                return get_xyxy_clz_con_emergency(infos[0])
            return None
            
    except Exception as e:
        logger.error(f"Emergency: parse_y_pred failed: {e}")
        return [] if islist else None

class Mini(nn.Module):
    class ConvBN(nn.Module):
        def __init__(self, cin, cout, kernel_size=3, stride=1, padding=None):
            super().__init__()
            padding = (kernel_size - 1) // 2 if not padding else padding
            self.conv = nn.Conv2d(cin, cout, kernel_size, stride, padding, bias=False)
            self.bn = nn.BatchNorm2d(cout, momentum=0.01)
            self.relu = nn.LeakyReLU(0.1, inplace=True)
        
        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))
    
    def __init__(self, anchors, class_types, inchennel=3):
        super().__init__()
        self.oceil = len(anchors) * (5 + len(class_types))
        self.model = nn.Sequential(
            OrderedDict([
                ('ConvBN_0', self.ConvBN(inchennel, 32)),
                ('Pool_0', nn.MaxPool2d(2, 2)),
                ('ConvBN_1', self.ConvBN(32, 48)),
                ('Pool_1', nn.MaxPool2d(2, 2)),
                ('ConvBN_2', self.ConvBN(48, 64)),
                ('Pool_2', nn.MaxPool2d(2, 2)),
                ('ConvBN_3', self.ConvBN(64, 80)),
                ('Pool_3', nn.MaxPool2d(2, 2)),
                ('ConvBN_4', self.ConvBN(80, 96)),
                ('Pool_4', nn.MaxPool2d(2, 2)),
                ('ConvBN_5', self.ConvBN(96, 102)),
                ('ConvEND', nn.Conv2d(102, self.oceil, 1)),
            ])
        )
    
    def forward(self, x):
        return self.model(x).permute(0, 2, 3, 1)

def get_clz_rect_from_image(image_data, state):
    """Emergency safe version"""
    try:
        if not state or 'net' not in state:
            return [], None
            
        net = state['net']
        anchors = state.get('anchors', [])
        class_types = state.get('class_types', {})
        
        if not anchors or not class_types:
            return [], None
            
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        npimg = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if npimg is None:
            return [], None
        
        height, width = npimg.shape[:2]
        npimg = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)
        npimg = cv2.resize(npimg, (416, 416), interpolation=cv2.INTER_LINEAR)
        npimg_ = np.transpose(npimg, (2,1,0))
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(npimg_).unsqueeze(0).to(DEVICE)
            if USE_CUDA:
                input_tensor = input_tensor.half()
            
            y_pred = net(input_tensor)
        
        v = parse_y_pred(y_pred, anchors, class_types, islist=True, threshold=0.2, nms_threshold=0.4)
        ret = []
        
        for i in v:
            if len(i) >= 4:
                rect, clz, con, log_cons = i[0], i[1], i[2], i[3]
                rw, rh = width/416, height/416
                rect[0] = int(rect[0]*rw)
                rect[2] = int(rect[2]*rw)
                rect[1] = int(rect[1]*rh)
                rect[3] = int(rect[3]*rh)
                ret.append([clz, rect])
                
        return ret, npimg
        
    except Exception as e:
        logger.error(f"Emergency: get_clz_rect_from_image failed: {e}")
        return [], None

def get_cut_img(npimg, rects):
    """Emergency safe version"""
    ret = []
    try:
        for item in rects:
            if len(item) >= 2:
                clz, rect = item[0], item[1]
                if len(rect) >= 4:
                    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(npimg.shape[1], x2), min(npimg.shape[0], y2)
                    if x2 > x1 and y2 > y1:
                        ret.append([clz, npimg[y1:y2,x1:x2,:], (x1,y1,x2,y2)])
    except:
        pass
    return ret

def get_flags_rects_from_image(image_data, state):
    """EMERGENCY ULTRA-SAFE VERSION - Will never crash"""
    try:
        if state is None:
            return None, None, None
            
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        s = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if s is None or s.size == 0:
            return None, None, None
        
        height, width = s.shape[:2]
        
        # Emergency safe slicing
        if height < 200 or width < 84:
            return None, None, None
            
        try:
            # Safer slicing with bounds checking
            end_height = min(height, s.shape[0])
            a = s[160:end_height, 0:min(22, width), :]
            b = s[160:end_height, 28:min(50, width), :]
            c = s[160:end_height, 56:min(78, width), :]
            
            if a.shape[0] < 40 or a.shape[1] < 20:
                return None, None, None
                
            a1 = a[40:min(60, a.shape[0]), :, :] if a.shape[0] > 40 else a
            a2 = a[0:min(20, a.shape[0]), :, :] if a.shape[0] > 0 else a
            b1 = b[40:min(60, b.shape[0]), :, :] if b.shape[0] > 40 else b
            b2 = b[0:min(20, b.shape[0]), :, :] if b.shape[0] > 0 else b
            c1 = c[40:min(60, c.shape[0]), :, :] if c.shape[0] > 40 else c
            c2 = c[0:min(20, c.shape[0]), :, :] if c.shape[0] > 0 else c
            
        except:
            return None, None, None
        
        def get_match_lens_emergency(i1, i2):
            """Ultra-safe matching"""
            try:
                if i1.size == 0 or i2.size == 0:
                    return 0
                    
                i1 = cv2.resize(i1, (min(i1.shape[1]*4, 800), min(i1.shape[0]*4, 600)), interpolation=cv2.INTER_LINEAR)
                i2 = cv2.resize(i2, (min(i2.shape[1]*2, 400), min(i2.shape[0]*2, 300)), interpolation=cv2.INTER_LINEAR)
                
                sift = get_sift_detector()
                kp1, des1 = sift.detectAndCompute(i1, None)
                kp2, des2 = sift.detectAndCompute(i2, None)
                
                if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
                    return 0
                    
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)
                good = 0
                
                for match_pair in matches:
                    if len(match_pair) >= 2:
                        m, n = match_pair[0], match_pair[1]
                        if m.distance <= 0.88 * n.distance:
                            good += 1
                            
                return good
                
            except:
                return 0
        
        def get_flag_rect_emergency(k12, cut_imgs, st):
            """Ultra-safe flag rect detection"""
            try:
                if len(k12) < 2:
                    return []
                    
                k1, k2 = k12[0], k12[1]
                r = []
                
                for item in cut_imgs:
                    if len(item) >= 3:
                        clz, npimg, rect = item[0], item[1], item[2]
                        if clz == '1':
                            r1 = get_match_lens_emergency(k1, npimg)
                            r.append([r1, rect, st])
                        elif clz == '2':
                            r2 = get_match_lens_emergency(k2, npimg)
                            r.append([r2, rect, st])
                            
                return sorted(r, key=lambda i: i[0]) if r else []
                
            except:
                return []
        
        try:
            rects, processed_img = get_clz_rect_from_image(image_data, state)
            if not rects:
                return None, None, None
                
            v = get_cut_img(s, rects)
            if len(v) == 0:
                return None, None, None
            
            rs1 = get_flag_rect_emergency([a1, a2], v, 1)
            rs2 = get_flag_rect_emergency([b1, b2], v, 2)
            rs3 = get_flag_rect_emergency([c1, c2], v, 3)
            rs = rs1 + rs2 + rs3
            
            if len(rs) < 3:
                return None, None, None
            
            # EMERGENCY RECT SELECTION - Will never crash
            r = []
            used_types = set()
            
            # Select best from each type
            for target_type in [1, 2, 3]:
                candidates = [x for x in rs if len(x) >= 3 and x[2] == target_type]
                if candidates:
                    best = max(candidates, key=lambda x: x[0])
                    r.append(best)
                    used_types.add(target_type)
                    
            if len(r) >= 3:
                r = sorted(r[:3], key=lambda x: x[2])
                # if DEBUG:
                #     logger.debug(f"Detected rects: {r}")
                return r[0][1], r[1][1], r[2][1]
            else:
                return None, None, None
                
        except Exception as e:
            logger.error(f"Emergency: processing failed: {e}")
            return None, None, None
            
    except Exception as e:
        logger.error(f"Emergency: get_flags_rects_from_image failed: {e}")
        return None, None, None

class Dun163:
    def __init__(self, id_, *, referer, fp_h, ua, thread_id, domain=None):
        self.fp = None
        self.resp_json2 = None
        self.domain = domain if domain else random.choice(DUN163_DOMAINS)
        self.thread_id = thread_id
        self._current_image_data = None
        self._current_rects = None
        self._current_click_points = None
        
        self.request_params = {
            'id': id_,
            'referer': referer,
            'fp_h': fp_h,
            'ua': ua
        }
        self.ss = self.set_session()
        
        self.ctx = get_compiled_js('dun163.js')
    
    def set_session(self):
        session = requests.Session()
        
        domain_host = self.domain.replace('https://', '').replace('http://', '')
        
        session.headers.update({
            "Accept": "*/*",
            "Accept-Language": "*",
            "Accept-Encoding": "*",
            "Accept-Post": "*/*",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Pragma": "no-cache",
            "Referer": self.request_params['referer'],
            "User-Agent": self.request_params['ua'],
            "Host": domain_host,
            "X-Forwarded-For": f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "X-Real-IP": f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        })
        
        session.timeout = (5, 10)
        return session
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def get_jsonp(text):
        try:
            jsonp_str = re.search(r"\((.*)\)", text, re.S)
            if jsonp_str:
                return json.loads(jsonp_str.group(1))
            return {}
        except:
            return {}
    
    @staticmethod
    def random_jsonp_str():
        s = string.ascii_lowercase + string.digits
        text = ''.join(random.choices(s, k=7))
        return "__JSONP_" + text + '_'
    
    def request_getconf(self):
        try:
            url = self.domain + '/api/v2/getconf'
            
            params = {
                "referer": self.request_params['referer'],
                "zoneId": "",
                "dt": "",
                "id": self.request_params['id'],
                "ipv6": "false",
                "runEnv": "10",
                "iv": "5",
                "loadVersion": "2.5.3",
                "lang": "en-US",
                "callback": self.random_jsonp_str() + '0'
            }
            
            response = self.ss.get(url, params=params)
            response.raise_for_status()
            resp_json = self.get_jsonp(response.text)
            
            return resp_json.get('data', {})
        except:
            return {}
    
    def request_get(self, dt, bid, ac_token, ir_token=None):
        try:
            url = self.domain + '/api/v3/get'
            
            fp = self.ctx.call('get_fp', self.request_params['fp_h'], self.request_params['ua'])
            cb = self.ctx.call('get_cb')
            
            self.fp = fp
            
            params = {
                "referer": self.request_params['referer'],
                "zoneId": "CN31",
                "dt": dt,
                "id": bid,
                "fp": fp,
                "https": "true",
                "type": "",
                "version": "2.28.5",
                "dpr": "1",
                "dev": "1",
                "cb": cb,
                "ipv6": "false",
                "runEnv": "10",
                "group": "",
                "scene": "",
                "lang": "en-US",
                "sdkVersion": "",
                "loadVersion": "2.5.3",
                "iv": "4",
                "user": "",
                "width": "320",
                "audio": "false",
                "sizeType": "10",
                "smsVersion": "v3",
                "token": "",
                "callback": self.random_jsonp_str() + '0'
            }
            
            if ir_token:
                params["irToken"] = ir_token
            
            resp_text = self.ss.get(url, params=params).text
            resp_json = self.get_jsonp(resp_text)
            
            return resp_json.get('data', {})
        except:
            return {}
    
    def request_check(self, dt, bid, *, token, captcha_type=7, click_data=None):
        try:
            url = self.domain + '/api/v3/check'
            
            js_start_time = time.time()
            
            if captcha_type == 7 and click_data:            
                check_data = self.ctx.call('get_click_check_data', click_data, token)
            else:
                check_data = '{"d":"","m":"","p":"","ext":""}'
            
            cb = self.ctx.call('get_cb')
            
            js_time = time.time() - js_start_time
            
            params = {
                "referer": self.request_params['referer'],
                "zoneId": "CN31", 
                "dt": dt,
                "id": bid,
                "token": token,
                "data": check_data,
                "width": "320",
                "type": str(captcha_type),
                "version": "2.28.5",
                "cb": cb,
                "user": "",
                "extraData": "",
                "bf": "0",
                "runEnv": "10",
                "sdkVersion": "",
                "loadVersion": "2.5.3", 
                "iv": "4",
                "callback": self.random_jsonp_str() + '1'
            }
            
            resp = self.ss.get(url, params=params)
            resp_json = self.get_jsonp(resp.text)
            
            return resp_json.get('data', {}), js_time
        except:
            return {}, 0.0
    
    def handle_click_captcha_hybrid(self, bg_url, token, attempt_num=0):
        """EMERGENCY SAFE VERSION"""
        try:
            headers = {"User-Agent": self.request_params['ua']}
            
            resp = requests.get(bg_url, headers=headers, timeout=10)
            resp.raise_for_status()
            
            image_data = resp.content
            img_start_time = time.time()
            
            state = get_global_model()
            if not state:
                return self.generate_emergency_clicks(), 0.0
            
            rects = get_flags_rects_from_image(image_data, state)
            img_time = time.time() - img_start_time
            
            self._current_image_data = image_data
            self._current_rects = rects
            
            # EMERGENCY SAFE RECT HANDLING
            rect1 = safe_list_access(rects, 0)
            rect2 = safe_list_access(rects, 1)
            rect3 = safe_list_access(rects, 2)
            
            if rect1 is not None and rect2 is not None and rect3 is not None:
                click_points = []
                
                for rect in [rect1, rect2, rect3]:
                    if rect and len(rect) >= 4:
                        x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        offset_x = random.randint(-1, 1)
                        offset_y = random.randint(-1, 1)
                        
                        final_x = max(5, min(center_x + offset_x, 315))
                        final_y = max(5, min(center_y + offset_y, 195))
                        
                        click_points.append({
                            "x": final_x,
                            "y": final_y
                        })
                
                if len(click_points) >= 3:
                    self._current_click_points = click_points[:3]
                    return click_points[:3], img_time
            
            # Emergency fallback
            click_points = self.generate_emergency_clicks()
            self._current_click_points = click_points
            return click_points, img_time
            
        except Exception as e:
            logger.error(f"Thread-{self.thread_id} | Emergency: handle_click_captcha_hybrid failed: {e}")
            return self.generate_emergency_clicks(), 0.0
    
    def generate_emergency_clicks(self):
        """Emergency click generator - Will never fail"""
        try:
            patterns = [
                [(80, 70), (160, 120), (240, 90)],
                [(70, 100), (160, 95), (250, 105)],
                [(160, 60), (110, 130), (210, 140)],
            ]
            
            selected_pattern = random.choice(patterns)
            click_points = []
            
            for x, y in selected_pattern:
                offset_x = random.randint(-5, 5)
                offset_y = random.randint(-5, 5)
                
                final_x = max(15, min(x + offset_x, 305))
                final_y = max(15, min(y + offset_y, 185))
                
                click_points.append({
                    "x": final_x,
                    "y": final_y
                })
            
            return click_points
        except:
            # Ultimate fallback
            return [{"x": 80, "y": 70}, {"x": 160, "y": 120}, {"x": 240, "y": 90}]
    
    def save_token_locally(self, validate_token):
        """Saves the generated token to a local file, using the requested format."""
        try:
            # Write only the token followed by a newline, removing all timestamps/thread info
            line = f"{validate_token}\n"
            
            with file_lock: # Use the global lock to ensure thread-safe writing
                with open(TOKEN_OUTPUT_FILE, 'a') as f:
                    f.write(line)
            
            return True
        except Exception as e:
            logger.error(f"T-{self.thread_id} | Local save error: {e}")
            return False
    
    def run(self, attempt_num=0):
        """EMERGENCY SAFE VERSION"""
        try:
            get_conf_data = self.request_getconf()
            if not get_conf_data:
                return False
                
            dt = get_conf_data.get('dt')
            ac_data = get_conf_data.get('ac', {})
            ac_token = ac_data.get('token')
            bid = ac_data.get('bid')
            
            ir_data = get_conf_data.get('ir', {})
            ir_token = ir_data.get('token') if ir_data.get('enable') else None
            
            get_data = self.request_get(dt, bid, ac_token, ir_token)
            if not get_data:
                return False
                
            captcha_type = get_data.get('type', 7)
            token = get_data.get('token')
            
            if not token:
                return False
            
            if captcha_type == 7:
                bg_urls = get_data.get('bg', [])
                if not bg_urls:
                    return False
                    
                click_points, img_time = self.handle_click_captcha_hybrid(bg_urls[0], token, attempt_num)
                resp_json, js_time = self.request_check(dt, bid, token=token, captcha_type=7, click_data=click_points)
            else:
                return False
            
            self.resp_json2 = resp_json
            
            if resp_json.get('result') == True:
                validate_raw = resp_json.get('validate', '')
                validate_decoded = ""
                
                if validate_raw and self.ctx:
                    try:
                        validate_decoded = self.ctx.call('do_onVerify', validate_raw, self.fp)
                    except:
                        pass
                
                if validate_decoded and len(validate_decoded.strip()) > 10:
                    # Call the new local saving function
                    local_success = self.save_token_locally(validate_decoded)
                    
                    if local_success:
                        # Updated success log
                        logger.success(f'T-{self.thread_id} SUCCESS: {validate_decoded} | Saved to {TOKEN_OUTPUT_FILE}')
                    else:
                        # Updated failure log
                        logger.warning(f'T-{self.thread_id} | IMG:{img_time:.1f}s | JS:{js_time:.1f}s | Token:{validate_decoded} | Local save: FAILED')

                    return True # We still return True because a token was generated
                else:
                    logger.warning(f'T-{self.thread_id} | Verification OK but invalid token')
                    return True
            else:
                return False
                
        except Exception as e:
            logger.error(f'T-{self.thread_id} | Emergency: run() failed: {str(e)[:100]}')
            return False

def worker_thread(thread_id, config):
    """EMERGENCY SAFE WORKER"""
    try:
        d = Dun163(
            id_=config['ID_'], 
            referer=config['REFERER'], 
            fp_h=config['FP_H'], 
            ua=config['UA'], 
            thread_id=thread_id, 
            domain=config['DOMAIN']
        )
        
        attempt = 0
        success_count = 0
        
        while True:
            attempt += 1
            try:
                time.sleep(random.uniform(0.5, 1.0))
                success = d.run(attempt_num=attempt)
                
                if success:
                    success_count += 1
                    logger.info(f"T-{thread_id} | Attempt {attempt} | Success #{success_count}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"T-{thread_id} | Error: {str(e)[:100]}")

    except Exception as e:
        logger.error(f"T-{thread_id} | Worker failed: {e}")

def main():
    logger.info("噫 Starting Solver Script...")
    
    model_state = initialize_global_model()
    if not model_state:
        logger.error("Model not available - cannot continue")
        return
        
    js_ctx = get_compiled_js('dun163.js')
    if not js_ctx:
        logger.error("JavaScript not available - cannot continue")
        return
    
    sift_detector = get_sift_detector()
    logger.success("All resources loaded")
    
    # Single config using bypasser-og.py settings
    config = {
        'ID_': ID,
        'REFERER': REFERER,
        'FP_H': FP_H,
        'UA': UserAgent().random,
        'DOMAIN': DUN163_DOMAINS[0]
    }
    
    NUM_THREADS = 5
    
    logger.info(f"櫨 Starting {NUM_THREADS} worker threads")
    logger.info(f"投 ID: {ID}")
    logger.info(f"投 REFERER: {REFERER}")
    logger.info(f"投 OUTPUT FILE: {TOKEN_OUTPUT_FILE}") # Updated log line
    logger.info("-" * 50)
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []
        
        for i in range(NUM_THREADS):
            # Create a copy with random UA for each thread
            thread_config = config.copy()
            thread_config['UA'] = UserAgent().random
            thread_config['DOMAIN'] = DUN163_DOMAINS[i % len(DUN163_DOMAINS)]
            future = executor.submit(worker_thread, i+1, thread_config)
            futures.append(future)
        
        try:
            for future in futures:
                future.result()
        except KeyboardInterrupt:
            logger.warning("Stopping...")
            executor.shutdown(wait=True)

if __name__ == '__main__':
    main()