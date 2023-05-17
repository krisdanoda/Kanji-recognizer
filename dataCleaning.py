from collections import Counter

def remove_min_occurences(labels, imgs):
    count = Counter(labels)
    filtered_labels = [x for x in labels if count[x] > 100]
    filtered_images = []
    for r in range(0,len(labels)):
        if count[labels[r]]>100:
            filtered_images.append(imgs[r])
    return filtered_labels, filtered_images


