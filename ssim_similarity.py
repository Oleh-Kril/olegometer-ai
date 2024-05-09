from skimage import metrics

def map_range(value, input_min, input_max, output_min, output_max):
    # Map the value from the input range to the output range
    return ((value - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min

def get_similarity(image1, image2):
    ssi_index, _ = metrics.structural_similarity(image1, image2, multichannel=True, full=True, data_range=1.0, channel_axis=2)

    ssi_similarity = map_range(ssi_index, -1, 1, 0, 1)

    return ssi_similarity