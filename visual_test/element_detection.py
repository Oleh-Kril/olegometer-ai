import cv2
import numpy as np
import matplotlib.pyplot as plt
from visual_test.image_utils import contains

def visualize_detection_step(gray, edges, edges_dil, contours, elems, title_prefix="", containers=None):
    n_dilations = len(edges_dil) if isinstance(edges_dil, list) else 1
    n_rows = 2 + (n_dilations - 1) // 3
    fig, ax = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    
    ax[0, 0].imshow(gray, cmap='gray')
    ax[0, 0].set_title(f'{title_prefix} Grayscale')
    
    ax[0, 1].imshow(edges, cmap='gray')
    ax[0, 1].set_title(f'{title_prefix} Canny Edges')
    
    if isinstance(edges_dil, list):
        for i, dil in enumerate(edges_dil):
            row = i // 3
            col = i % 3
            ax[row, col].imshow(dil, cmap='gray')
            ax[row, col].set_title(f'{title_prefix} Dilated Edges (k={i+1})')
    else:
        ax[0, 2].imshow(edges_dil, cmap='gray')
        ax[0, 2].set_title(f'{title_prefix} Dilated Edges')
    
    ax[-2, 0].imshow(gray, cmap='gray')
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax[-2, 0].add_patch(rect)
    ax[-2, 0].set_title(f'{title_prefix} Contours')
    
    ax[-2, 1].imshow(gray, cmap='gray')
    for elem in elems:
        x, y, w, h = elem['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='blue', linewidth=1)
        ax[-2, 1].add_patch(rect)
    ax[-2, 1].set_title(f'{title_prefix} All Elements')
    
    ax[-2, 2].imshow(gray, cmap='gray')
    if title_prefix.startswith('Atomics') and containers is not None:
        for elem in elems:
            x, y, w, h = elem['bbox']
            inside = any(contains(cont['bbox'], elem['bbox']) for cont in containers)
            color = 'orange' if inside else 'green'
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2)
            ax[-2, 2].add_patch(rect)
        for cont in containers:
            x, y, w, h = cont['bbox']
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2, linestyle='--')
            ax[-2, 2].add_patch(rect)
        ax[-2, 2].set_title(f'{title_prefix}\nGreen=Free, Orange=Child, Red=Container')
    else:
        for elem in elems:
            x, y, w, h = elem['bbox']
            color = 'red' if elem.get('children') else 'green'
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1)
            ax[-2, 2].add_patch(rect)
        ax[-2, 2].set_title(f'{title_prefix} Hierarchy\nRed=Container, Green=Atomic')
    
    for i in range(n_rows):
        for j in range(3):
            if i == 0 and j == 2 and isinstance(edges_dil, list):
                continue
            if i >= 1 and i < n_rows - 2:
                continue
            if i == n_rows - 1 and j >= 0:
                ax[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

def build_hierarchy_bbox(contours, min_bbox_area):
    elements = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_bbox_area:
            continue
        elements.append({
            'bbox': [x, y, w, h],
            'contour': c,
            'children': []
        })

    if not elements:
        return []

    elements.sort(key=lambda e: e['bbox'][2] * e['bbox'][3])

    for idx, elem in enumerate(elements):
        for cand in elements[idx + 1:]:
            if contains(cand['bbox'], elem['bbox']):
                cand['children'].append(elem)
                break   # stop at the *closest* container

    return elements

def _merge_levels(levels):
    all_elements = [elem for lvl in levels for elem in lvl]

    all_elements.sort(key=lambda e: e['bbox'][2] * e['bbox'][3])

    for idx, elem in enumerate(all_elements):
        for cand in all_elements[idx + 1:]:
            if contains(cand['bbox'], elem['bbox']):
                if 'children' not in cand:
                    cand['children'] = []
                cand['children'].append(elem)
                break

    return all_elements


def detect_elements_multiscale(gray, const, mask=None):
    min_bbox_area = const.get('MIN_BBOX_AREA', 1)
    edges = cv2.Canny(
        gray,
        const['CANNY_LOW_THRESHOLD'],
        const['CANNY_HIGH_THRESHOLD']
    )
    if mask is not None:
        edges[mask == 0] = 0

    kernel_sizes = const.get('EDGE_DILATION_SIZES', [2, 20])

    level_elems = []
    vis_edges = []

    for k in kernel_sizes:
        kernel = np.ones((k, k), np.uint8)
        edges_dil = cv2.dilate(edges, kernel, iterations=1)
        vis_edges.append(edges_dil)
        contours, _ = cv2.findContours(
            edges_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        elems_lvl = build_hierarchy_bbox(contours, min_bbox_area)
        level_elems.append(elems_lvl)

    all_elements = _merge_levels(level_elems)
    containers = [e for e in all_elements if e['children']]
    atomics = [e for e in all_elements if not e['children']]

    # visualize_detection_step(
    #     gray, edges, vis_edges, [], all_elements,
    #     "Multiscale Detection", containers
    # )
    return all_elements, containers, atomics, edges, vis_edges
