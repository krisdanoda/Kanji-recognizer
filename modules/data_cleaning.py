from collections import Counter
import cv2

def remove_min_occurences(labels, imgs):
    count = Counter(labels)
    filtered_labels = [x for x in labels if count[x] > 100]
    filtered_images = []
    for r in range(0,len(labels)):
        if count[labels[r]]>100:
            filtered_images.append(imgs[r])
    return filtered_labels, filtered_images

def remove_by_contours(labels, images):
    full_count = 0
    image_contours = {}
    filtered_images = []
    filtered_labels = []
    label = labels[0]
    for i in range(0,len(images)):
        count = 0
        if label != labels[i]:
            full_count = full_count/len(image_contours)
            for k in image_contours.keys():
                if image_contours[k] > full_count-3 and image_contours[k] < full_count+3:
                    filtered_images.append(images[k])
                    filtered_labels.append(labels[k])
            full_count = 0
            image_contours = {}
        img = images[i]
        ret, thresh = cv2.threshold(img, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for c in contours:
            if cv2.contourArea(c) > 15:
                count+=1
                full_count+=1
        image_contours[i] = count
        label = labels[i]
    return filtered_labels, filtered_images

def get_removed_contours(labels, images):
    full_count = 0
    image_contours = {}
    filtered_images = []
    filtered_labels = []
    label = labels[0]
    for i in range(0,len(images)):
        count = 0
        if label != labels[i]:
            full_count = full_count/len(image_contours)
            for k in image_contours.keys():
                if image_contours[k] < full_count-3 or image_contours[k] > full_count+3:
                    filtered_images.append(images[k])
                    filtered_labels.append(labels[k])
            full_count = 0
            image_contours = {}
        img = images[i]
        ret, thresh = cv2.threshold(img, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for c in contours:
            if cv2.contourArea(c) > 15:
                count+=1
                full_count+=1
        image_contours[i] = count
        label = labels[i]
    return filtered_labels, filtered_images
