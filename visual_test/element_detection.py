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

def _are_elements_similar_height(elem1, elem2, threshold):
    """Check if two elements have similar heights within the threshold percentage."""
    h1 = elem1['bbox'][3]
    h2 = elem2['bbox'][3]
    max_h = max(h1, h2)
    height_diff_percent = abs(h1 - h2) / max_h
    return height_diff_percent <= threshold

def _are_elements_aligned_horizontally(elem1, elem2):
    """
    Check if a horizontal line from the middle of the first element passes through the second element.
    """
    # Get the y-coordinate of the middle point of the first element
    y_line = elem1['bbox'][1] + elem1['bbox'][3] / 2
    
    # Get the y-coordinates of the second element's top and bottom
    y2_top = elem2['bbox'][1]
    y2_bottom = elem2['bbox'][1] + elem2['bbox'][3]
    
    # Check if the line passes through the second element
    return y2_top <= y_line <= y2_bottom

def _create_wrapping_container(elems):
    """
    Return the minimal axis-aligned bbox that **strictly** contains every child.
    We compute it in one pass and then assert containment.
    """
    if not elems:
        return None

    xs      = [e['bbox'][0]           for e in elems]
    ys      = [e['bbox'][1]           for e in elems]
    rights  = [e['bbox'][0] + e['bbox'][2] for e in elems]
    bottoms = [e['bbox'][1] + e['bbox'][3] for e in elems]

    x, y     = min(xs),      min(ys)
    right    = max(rights)
    bottom   = max(bottoms)
    width    = right  - x
    height   = bottom - y

    container = {
        'bbox':    [x, y, width, height],
        'children': elems,
        'contour': None
    }

    # Sanity-check that every child really is inside—no off-by-ones allowed
    for e in elems:
        if not contains(container['bbox'], e['bbox']):
            cx, cy, cw, ch = container['bbox']
            ex, ey, ew, eh = e['bbox']
            raise RuntimeError(
                f"Child {e['bbox']} not in container {container['bbox']}"
            )

    return container

def _find_existing_container(elem, containers):
    """Find if an element is already wrapped in a container."""
    for container in containers:
        if container.get('children') and elem in container['children']:
            return container
    return None

def _are_elements_in_same_container(elems, containers):
    """Check if all elements are already wrapped in the same container."""
    if not elems:
        return False
    
    # Find container for first element
    first_container = _find_existing_container(elems[0], containers)
    if not first_container:
        return False
    
    # Check if all other elements are in the same container
    return all(_find_existing_container(elem, containers) == first_container for elem in elems[1:])

def _are_elements_within_distance(elem1, elem2, distance_threshold):
    """
    Check if the distance between two elements is less than the threshold percentage
    of the width of the larger element.
    """
    # Get the right edge of the left element and left edge of the right element
    right_edge1 = elem1['bbox'][0] + elem1['bbox'][2]
    left_edge2 = elem2['bbox'][0]
    
    # Calculate the distance between elements
    distance = left_edge2 - right_edge1
    
    # Get the width of the larger element
    max_width = max(elem1['bbox'][2], elem2['bbox'][2])
    
    # Calculate the maximum allowed distance based on threshold percentage
    max_allowed_distance = max_width * (distance_threshold / 100)
    
    return distance <= max_allowed_distance

def wrap_similar_elements(elements, threshold, distance_threshold):
    """Wrap elements that are similar in height and aligned horizontally."""
    if not elements:
        return elements
    
    # Step 1: Create groups based on alignment and height similarity
    groups = []
    used_indices = set()  # Track indices instead of elements
    
    # Sort elements by x coordinate for consistent grouping
    sorted_elements = sorted(elements, key=lambda e: e['bbox'][0])
    
    for i, elem in enumerate(sorted_elements):
        if i in used_indices:
            continue
            
        current_group = [elem]
        used_indices.add(i)
        
        # Find all elements that align with current element
        for j, other_elem in enumerate(sorted_elements[i+1:], start=i+1):
            if j in used_indices:
                continue
                
            # Check if element should be added to current group
            should_add = True
            for group_elem in current_group:
                if not (_are_elements_similar_height(other_elem, group_elem, threshold) and 
                       _are_elements_aligned_horizontally(other_elem, group_elem)):
                    should_add = False
                    break
            
            if should_add:
                current_group.append(other_elem)
                used_indices.add(j)
        
        if current_group:
            groups.append(current_group)
    
    wrapped_elements = []
    containers = [e for e in elements if e.get('children')]
    
    for group in groups:
        wrapped_elements.extend(group)
        
        # Check if elements are close enough to be wrapped
        should_wrap = len(group) >= 2 and not _are_elements_in_same_container(group, containers)
        if should_wrap:
            # Sort group by x coordinate to ensure we check neighbors in order
            sorted_group = sorted(group, key=lambda e: e['bbox'][0])
            # Check distance between each pair of neighboring elements
            for i in range(len(sorted_group) - 1):
                elem1 = sorted_group[i]
                elem2 = sorted_group[i + 1]
                # Get the right edge of the left element and left edge of the right element
                right_edge1 = elem1['bbox'][0] + elem1['bbox'][2]
                left_edge2 = elem2['bbox'][0]
                # Calculate the distance between elements
                distance = left_edge2 - right_edge1
                # Get the width of the larger element
                max_width = max(elem1['bbox'][2], elem2['bbox'][2])
                # Calculate the maximum allowed distance based on threshold percentage
                max_allowed_distance = max_width * (distance_threshold / 100)
                if distance > max_allowed_distance:
                    should_wrap = False
                    break
            
            if should_wrap:
                container = _create_wrapping_container(group)
                wrapped_elements.append(container)
    
    return wrapped_elements

def _remove_duplicate_bboxes(elements):
    """
    Remove elements that have the same bbox, keeping only the first occurrence.
    """
    seen_bboxes = set()
    unique_elements = []
    
    for elem in elements:
        bbox_tuple = tuple(elem['bbox'])
        if bbox_tuple not in seen_bboxes:
            seen_bboxes.add(bbox_tuple)
            unique_elements.append(elem)
    
    return unique_elements

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
    
    # Apply wrapping for similar elements
    wrap_threshold = const.get('WRAP_CONTAINER_SIZE_THRESHOLD', 0.2)
    all_elements = wrap_similar_elements(all_elements, wrap_threshold, const.get('WRAP_CONTAINER_DISTANCE_THRESHOLD', 0.2))
    
    # Remove elements with duplicate bboxes
    all_elements = _remove_duplicate_bboxes(all_elements)
    
    containers = [e for e in all_elements if e.get('children')]
    atomics = [e for e in all_elements if not e.get('children')]

    # visualize_detection_step(
    #     gray, edges, vis_edges, [], all_elements,
    #     "Multiscale Detection", containers
    # )
    return all_elements, containers, atomics, edges, vis_edges
