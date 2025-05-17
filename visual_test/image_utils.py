import numpy as np
from PIL import Image
import imagehash

def contains(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return bx >= ax and by >= ay and bx + bw <= ax + aw and by + bh <= ay + ah

def phash_region(gray, bbox):
    x, y, w, h = bbox
    roi = gray[y:y+h, x:x+w]
    if w == 0 or h == 0:
        return imagehash.ImageHash(np.zeros((32, 32), dtype=np.uint8))

    pil_image = Image.fromarray(roi)
    
    pil_image = pil_image.resize((32, 32), Image.Resampling.LANCZOS)
    
    return imagehash.phash(pil_image)