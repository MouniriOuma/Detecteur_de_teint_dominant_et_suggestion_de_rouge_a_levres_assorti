import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import pprint
from matplotlib import pyplot as plt


def extractSkin(image):
    # Taking a copy of the image
    img = image.copy()
    # Converting from BGR Colours Space to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining HSV Threadholds
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    # Return the Skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


def removeBlack(estimator_labels, estimator_cluster):

    # Check for black
    hasBlack = False

    # Get the total number of occurance for each color
    occurance_counter = Counter(estimator_labels)

    # Quick lambda function to compare to lists
    def compare(x, y): return Counter(x) == Counter(y)

    # Loop through the most common occuring color
    for x in occurance_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0,0,0] that if it is black
        if compare(color, [0, 0, 0]) == True:
            # delete the occurance
            del occurance_counter[x[0]]
            # remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break

    return (occurance_counter, estimator_cluster, hasBlack)


def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):

    # Variable to keep count of the occurance of each color predicted
    occurance_counter = None

    # Output list variable to return
    colorInformation = []

    # Check for Black
    hasBlack = False

    # If a mask has be applied, remove th black
    if hasThresholding == True:

        (occurance, cluster, black) = removeBlack(
            estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black

    else:
        occurance_counter = Counter(estimator_labels)

    # Get the total sum of all the predicted occurances
    totalOccurance = sum(occurance_counter.values())

    # Loop through all the predicted colors
    for x in occurance_counter.most_common(len(estimator_cluster)):

        index = (int(x[0]))

        # Quick fix for index out of bound when there is no threshold
        index = (index-1) if ((hasThresholding & hasBlack)
                              & (int(index) != 0)) else index

        # Get the color number into a list
        color = estimator_cluster[index].tolist()

        # Get the percentage of each color
        color_percentage = (x[1]/totalOccurance)

        # make the dictionay of the information
        colorInfo = {"cluster_index": index, "color": color,
                     "color_percentage": color_percentage}

        # Add the dictionary to the list
        colorInformation.append(colorInfo)

    return colorInformation


def extractDominantColor(image, number_of_colors=3, hasThresholding=False):

    # Quick Fix Increase cluster counter to neglect the black(Read Article)
    if hasThresholding == True:
        number_of_colors += 1

    # Taking Copy of the image
    img = image.copy()

    # Convert Image into RGB Colours Space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape Image
    img = img.reshape((img.shape[0]*img.shape[1]), 3)

    # Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)

    # Fit the image
    estimator.fit(img)

    # Get Colour Information
    colorInformation = getColorInformation(
        estimator.labels_, estimator.cluster_centers_, hasThresholding)
    return colorInformation


def plotColorBar(colorInformation):
    # Create a 500x100 black image
    color_bar = np.zeros((100, 500, 3), dtype="uint8")

    top_x = 0
    for x in colorInformation:
        bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

        color = tuple(map(int, (x['color'])))

        cv2.rectangle(color_bar, (int(top_x), 0),
                      (int(bottom_x), color_bar.shape[0]), color, -1)
        top_x = bottom_x
    return color_bar


def choose_lipstick_color(dominant_colors):
    # Define lipstick color ranges based on skin color characteristics
    lipstick_colors = {
        'fair': [
            '#FF7F50',  # Coral (reddish-orange)
            '#FFE4C4',  # Bisque (nude)
            '#D2B48C',  # Tan (light brown)
            '#A0522D'  # Sienna (reddish-brown)
        ],
        'medium': [
            '#CC0000',  # Brick red
            '#FFEBCD',  # Blanched almond (nude)
            '#C08863',  # Sandy brown
            '#8B4513'  # Saddle brown
        ],
        'dark': [
            '#800000',  # Maroon
            '#FFE8D6',  # Peach puff (nude)
            '#996515',  # Bronze
            '#4A3B32'  # Deep brown
        ]
    }

    # Extract hue, saturation, and brightness values from dominant_colors
    hues = [color['color'][0] for color in dominant_colors]
    saturations = [color['color'][1] for color in dominant_colors]
    brightnesses = [color['color'][2] for color in dominant_colors]

    # Calculate average hue, saturation, and brightness values
    avg_hue = sum(hues) / len(dominant_colors)
    avg_saturation = sum(saturations) / len(dominant_colors)
    avg_brightness = sum(brightnesses) / len(dominant_colors)

    # Determine the skin tone based on the provided skin color characteristics
    if avg_brightness > 80:  # Adjusted brightness threshold for dark skin
        if avg_saturation < 100:
            skin_tone = 'fair'
        elif avg_saturation < 150:
            skin_tone = 'medium'
        else:
            skin_tone = 'dark'
    else:
        if avg_saturation < 100:
            skin_tone = 'fair'
        elif avg_saturation < 150:
            skin_tone = 'medium'
        else:
            skin_tone = 'dark'

    # Return suitable lipstick colors based on the determined skin tone
    if skin_tone:
        return lipstick_colors[skin_tone]
    else:
        return ['neutral', 'safe options']  # Default options if skin tone not recognized



"""## Section Two.4.2 : Putting it All together: Pretty Print
The function makes print out the color information in a readable manner
"""


def prety_print_data(color_info):
    for x in color_info:
        print(pprint.pformat(x))
        print()


"""
The below lines of code, is the implementation of the above defined function.
"""

'''
Skin Image Primary : https://raw.githubusercontent.com/octalpixel/Skin-Extraction-from-Image-and-Finding-Dominant-Color/master/82764696-open-palm-hand-gesture-of-male-hand_image_from_123rf.com.jpg
Skin Image light     : https://st4.depositphotos.com/6903990/27898/i/450/depositphotos_278981062-stock-photo-beautiful-young-woman-clean-fresh.jpg
Skin Image medium     : https://www.shutterstock.com/image-photo/portrait-young-beautiful-woman-perfect-600nw-2228044161.jpg
Skin Image dark   : https://static.ffx.io/images/$zoom_0.172%2C$multiply_0.7725%2C$ratio_1.5%2C$width_756%2C$x_0%2C$y_0/t_crop_custom/q_86%2Cf_auto/1f0aa1cbbf4da2e45e5112679d460854e882d5d7
'''


# Get Image from URL. If you want to upload an image file and use that comment the below code and replace with  image=cv2.imread("FILE_NAME")
image = imutils.url_to_image(
    "https://st4.depositphotos.com/6903990/27898/i/450/depositphotos_278981062-stock-photo-beautiful-young-woman-clean-fresh.jpg")

# Resize image to a width of 250
image = imutils.resize(image, width=250)

# Create subplots for displaying different information
plt.figure(figsize=(16, 8))

# Show image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
# plt.show()

# Apply Skin Mask
skin = extractSkin(image)



plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(skin, cv2.COLOR_BGR2RGB))
plt.title("Thresholded  Image")
# plt.show()

# Find the dominant color. Default is 1 , pass the parameter 'number_of_colors=N' where N is the specified number of colors
dominantColors = extractDominantColor(skin, hasThresholding=True)

# Show in the dominant color information
print("Color Information")
prety_print_data(dominantColors)

# Show in the dominant color as bar
print("Color Bar")
colour_bar = plotColorBar(dominantColors)
plt.subplot(2, 2, 3)
plt.axis("off")
plt.imshow(colour_bar)
plt.title("Color Bar")

#show lipstick choice
lipstick_suggestions = choose_lipstick_color(dominantColors)

# Show in the lipstick color information
print("lipstick :")
prety_print_data(lipstick_suggestions)

# Create a bar chart to visualize lipstick colors
plt.subplot(2, 2, 4)
colors = lipstick_suggestions
plt.barh(range(len(lipstick_suggestions)), [1]*len(lipstick_suggestions), color=colors)
plt.yticks(range(len(lipstick_suggestions)), lipstick_suggestions)
plt.title('Lipstick Color Suggestions')

plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()