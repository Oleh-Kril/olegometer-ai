import easyocr
from visual_test.custom_types import DiffType
from visual_test.image_utils import contains

def filter_nested_missing_elements(diffs):
    missing = [d for d in diffs if d['type'] == 'ELEMENT_MISSING_ON_WEBSITE_PAGE']
    keep = []
    for i, d in enumerate(missing):
        x1, y1, w1, h1 = d['bbox']
        contained = False
        for j, other in enumerate(missing):
            if i == j:
                continue
            x2, y2, w2, h2 = other['bbox']
            if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
                contained = True
                break
        if not contained:
            keep.append(d)
    return keep + [d for d in diffs if d['type'] != 'ELEMENT_MISSING_ON_WEBSITE_PAGE']

def closest(b, cont, axis, w_img, h_img):
    if cont is None:
        return min(b[0], w_img - (b[0] + b[2])) if axis == 'x' else min(b[1], h_img - (b[1] + b[3]))
    px, py, pw, ph = cont
    return min(b[0] - px, px + pw - (b[0] + b[2])) if axis == 'x' else min(b[1] - py, py + ph - (b[1] + b[3]))

# in visual_test/comparison.py

def compare_positions(eD, eR, parent_D, parent_R, masked_D, masked_R, constants, wrong_pos_seen):
    """
    Compare element positions, now relative to each side’s container.
    parent_D: bbox of the container in the design image (or None)
    parent_R: bbox of the container in the real image   (or None)
    """
    dx = eR['bbox'][0] - eD['bbox'][0]
    dy = eR['bbox'][1] - eD['bbox'][1]
    sx = abs(dx) > constants['POSITION_THRESHOLD']
    sy = abs(dy) > constants['POSITION_THRESHOLD']

    # no significant move → no diff
    if not (sx or sy):
        return None

    # avoid duplicate reporting per element
    key = tuple(eD['bbox'])
    if key in wrong_pos_seen:
        return None
    wrong_pos_seen.add(key)

    rec = {
        'bbox': eR['bbox'],
        'type': DiffType.WRONG_POSITION_WITHIN if parent_D else DiffType.WRONG_POSITION
    }

    # helper to pick distances against the correct container
    def dist(b, cont, axis, img_w, img_h):
        if cont is None:
            return min(b[0], img_w - (b[0] + b[2])) if axis=='x' else \
                   min(b[1], img_h - (b[1] + b[3]))
        px, py, pw, ph = cont
        if axis == 'x':
            return min(b[0] - px, px+pw - (b[0]+b[2]))
        else:
            return min(b[1] - py, py+ph - (b[1]+b[3]))

    if sx:
        rec['OriginalDistanceX'] = dist(eD['bbox'], parent_D,
                                        'x', masked_D.shape[1], masked_D.shape[0])
        rec['CurrentDistanceX']  = dist(eR['bbox'], parent_R,
                                        'x', masked_R.shape[1], masked_R.shape[0])
    if sy:
        rec['OriginalDistanceY'] = dist(eD['bbox'], parent_D,
                                        'y', masked_D.shape[1], masked_D.shape[0])
        rec['CurrentDistanceY']  = dist(eR['bbox'], parent_R,
                                        'y', masked_R.shape[1], masked_R.shape[0])

    return rec


def compare_sizes(eD, eR, constants):
    w0, h0 = eD['bbox'][2:4]
    w1, h1 = eR['bbox'][2:4]
    
    if abs(w0 - w1) > constants['SIZE_THRESHOLD'] or abs(h0 - h1) > constants['SIZE_THRESHOLD']:
        return {
            'type': DiffType.WRONG_SIZE,
            'bbox': eR['bbox'],
            'OriginalWidth': w0,
            'CurrentWidth': w1,
            'OriginalHeight': h0,
            'CurrentHeight': h1
        }
    return None

def compare_text(eD, eR, img_D, img_R):
    # reader = easyocr.Reader(['en', 'uk'], gpu=False)
    #
    # x0, y0, w0, h0 = eD['bbox']
    # x1, y1, w1, h1 = eR['bbox']
    #
    # tD = " ".join(reader.readtext(img_D[y0:y0+h0, x0:x0+w0], detail=0)).strip()
    # tR = " ".join(reader.readtext(img_R[y1:y1+h1, x1:x1+w1], detail=0)).strip()
    #
    # if tD and tR and tD != tR:
    #     return {
    #         'type': DiffType.WRONG_COPY,
    #         'bbox': eR['bbox'],
    #         'OriginalCopy': tD,
    #         'CurrentCopy': tR
    #     }
    return None 