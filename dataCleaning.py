from collections import Counter

def remove_min_occurences(labels, imgs):
    filtered_labels = [x for x in labels if count[x] > 200]
    count = Counter(labels)
    filtered_images = []
    for r in range(0,len(labels)):
        if(count[labels[r]]>200):
            filtered_images.append(imgs[r])
    return filtered_labels, filtered_images


