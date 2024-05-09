def edge_based_comparison(bbox1, bbox2, edges1, edges2, resize_h, resize_w):
    # Extract regions from edge maps based on bounding boxes
    region1 = edges1[bbox1[0]:bbox1[2], bbox1[1]:bbox1[3]]
    region2 = edges2[bbox2[0]:bbox2[2], bbox2[1]:bbox2[3]]

    resized_region_1 = resize(region1, (resize_h, resize_w), mode='constant', anti_aliasing=False)
    resized_region_2 = resize(region2, (resize_h, resize_w), mode='constant', anti_aliasing=False)
    # show_two_images(resized_region_1, resized_region_2)

    # Calculate Jaccard similarity
    intersection = np.logical_and(resized_region_1, resized_region_2).sum()
    union = np.logical_or(resized_region_1, resized_region_2).sum()

    # Avoid division by zero
    similarity = intersection / union if union > 0 else 0.0

    return similarity / (resize_h * resize_w) * 1e6


similarity = edge_based_comparison(bbox_of_original, bbox_of_to_compare, edges_original, edges_to_compare, resize_h,
                                   resize_w)
