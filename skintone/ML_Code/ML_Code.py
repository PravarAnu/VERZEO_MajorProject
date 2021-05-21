import os
import pprint
import sys
import cv2
import imutils
import numpy
from collections import Counter
from matplotlib import pyplot
from sklearn.cluster import KMeans



class FaceCropper(object):

    CASCADE_PATH = "data/haarcascades/haarcascade_frontalface_default.xml"
    #CASCADE_PATH = "data/haarcascades/haarcascade_frontalface_alt.xml"
    #CASCADE_PATH = "data/haarcascades/haarcascade_frontalface_alt2.xml"
    #CASCADE_PATH = "data/lbpcascades/lbpcascade_frontalface.xml"
    #CASCADE_PATH = "data/lbpcascades/lbpcascade_frontalface_improved.xml"

    def __init__(self):

        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, image_path, flag):

        img1 = cv2.imread(image_path)

        if (img1 is None):
            
            try:
                
                img2 = imutils.url_to_image(image_path)
                
            except:
                
                print()
                print("Image doesn't exist!!!")
                print("Please Try Again!!!")
                flag = 1
                return (image_path, flag)
            
            if (img2 is None):
                
                print()
                print("Image doesn't exist!!!")
                print("Please Try Again!!!")
                flag = 2
                return (image_path, flag)
            
            else:
                
                img = img2
        else:
            
            img = img1

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Show Original Image
        '''
        pyplot.figure(0)
        pyplot.subplot(1, 3, 1)
        pyplot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pyplot.title("Original Image")
        #pyplot.show()
        ''';
        
        faces = self.face_cascade.detectMultiScale(img, 1.1, 6)#, minSize = (100, 100))

        if (faces is None):
            
            print()
            print("No Face detected")
            print("Please Try Again!!!")
            flag = 3
            return (image_path, flag)
        
        for (x, y, w, h) in faces:
            
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 225, 225), 1)
            #break
        
        cv2.imshow('img', img)
        cv2.waitKey(1)
        cv2.destroyAllWindows()

        facecount = len(faces)
        #print("Detected faces: %d" % facecount)
        if facecount == 0:
            
            flag = 4
            return (image_path, flag)
        
        #height, width = img.shape[:2]
        
        i = 0

        for (x, y, w, h) in faces:

            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            #faceimg = img[ny:ny+nr, nx:nx+nr]
            #lastimg = cv2.resize(faceimg, (300, 300))
            lastimg = img[ny:ny+nr, nx:nx+nr]
            #image = imutils.resize(image, width = 300)
            
            cv2.imwrite("CROP%d.jpg" %i, lastimg)
            i += 1
            #break
        
        flag = 0
        return (image_path, flag)


def extractSkin(image):

    # Taking a copy of the image
    img = image.copy()

    # Converting from BGR Colours Space to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining HSV Threadholds
    lower_threshold = numpy.array([0, 48, 80], dtype = numpy.uint8)
    
    upper_threshold = numpy.array([20, 255, 255], dtype = numpy.uint8)

    # Single Channel mask, denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask = skinMask)

    # Return the Skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


def removeBlack(estimator_labels, estimator_cluster):

    # Check for black
    hasBlack = False

    # Get the total number of occurences for each color
    occurance_counter = Counter(estimator_labels)

    # Quick lambda function to compare to lists
    def compare(x, y): 
        return Counter(x) == Counter(y)

    # Loop through the most commonly occuring color
    for x in occurance_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0, 0, 0]/[1, 0, 0] that if it is black
        if compare(color, [0, 0, 0]) == True or compare(color, [1, 0, 0]) == True:
            
            # Delete the occurence
            del occurance_counter[x[0]]
            
            # Remove the cluster
            hasBlack = True
            estimator_cluster = numpy.delete(estimator_cluster, x[0], 0)
            break

    return (occurance_counter, estimator_cluster, hasBlack)


def getColorInformation(estimator_labels, estimator_cluster, hasThresholding = False):

    # Variable to keep count of the occurance of each color predicted
    occurance_counter = None

    # Output list variable to be returned
    colorInformation = []

    # Check for Black
    hasBlack = False

    # If a mask has be applied, remove the black
    if hasThresholding == True:

        (occurance, cluster, black) = removeBlack(estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black

    else:

        occurance_counter = Counter(estimator_labels)

    # Get the total sum of all the predicted occurances
    totalOccurance = sum(occurance_counter.values())

    # Loop through all the predicted colors
    for x in occurance_counter.most_common(len(estimator_cluster)):

        index = int(x[0])

        # Quick fix for index out of bound when there is no threshold
        index = (index - 1) if (hasThresholding & hasBlack & (int(index) != 0)) else index

        # Get the color number into a list
        color = estimator_cluster[index].tolist()

        # Get the percentage of each color
        color_percentage = (x[1] * 100 / totalOccurance)

        # Make dictionay of information
        colorInfo = {"cluster_index" : index, "color" : color, "color_percentage" : color_percentage}

        # Add the dictionary to the list
        colorInformation.append(colorInfo)

    return colorInformation


def extractDominantColor(image, number_of_colors = 5, hasThresholding = False):

    # Quick Fix : Increase cluster counter to neglect the black
    if hasThresholding == True:
        
        number_of_colors += 1

    # Taking Copy of the image
    img = image.copy()

    # Convert Image into RGB Colours Space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape Image
    img = img.reshape((img.shape[0] * img.shape[1]), 3)

    # Initiate KMeans Object
    estimator = KMeans(n_clusters = number_of_colors, random_state = 0)

    # Fit the image
    estimator.fit(img)

    # Get Colour Information
    colorInformation = getColorInformation(estimator.labels_, estimator.cluster_centers_, hasThresholding)
    
    return colorInformation


def plotColorBar(colorInformation):

    # Create a 500x100 black image
    color_bar = numpy.zeros((100, 500, 3), dtype = "uint8")

    top_x = 0

    for x in colorInformation:
        
        bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1] / 100)

        color = tuple(map(int, (x['color'])))

        cv2.rectangle(color_bar, (int(top_x), 0), (int(bottom_x), color_bar.shape[0]), color, -1)
        
        top_x = bottom_x

    return color_bar



##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################



# Taking Input   
detecter = FaceCropper()
(htap, galf) = detecter.generate(image_path = input("Image Address : "), flag = 0)
#print(htap)
#print(galf)

##################################################################################################################

# Cropped Image Doesn't Exist
if galf == 4:
    
    print("No Face Detected!!!")
    print("Please Try Again!!!")


# Cropped Image Exists
if galf == 0:

    # Reading Cropped Image
    image = cv2.imread("CROP0.jpg")    
    os.remove("CROP0.jpg")

    # Apply Skin Mask
    skin = extractSkin(image)
    #skin = extractSkin(cv2.resize(image, (300, 300)))
    
    # Find the dominant color
    # Default is 1 , pass the parameter 'number_of_colors = N' where N is the specified number of colors
    # hasThresholding = True ignores black
    dominantColors = extractDominantColor(skin, number_of_colors = 4, hasThresholding = True)

    
if galf == 0:
    
    # Show Cropped Image
    '''
    pyplot.subplot(1, 3, 2)
    pyplot.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pyplot.title("Cropped Image")
    #pyplot.show()
    ''';
    
    # Show Skin Mask
    '''
    pyplot.subplot(1, 3, 3)
    pyplot.imshow(cv2.cvtColor(skin, cv2.COLOR_BGR2RGB))
    pyplot.title("Thresholded Image")
    pyplot.tight_layout()
    #pyplot.show()
    ''';

##################################################################################################################    
##################################################################################################################
    
if galf == 0:  
    
    # Dominant Color Information as Bar Graph    
    '''
    #print("Color Bar")
    color_bar = plotColorBar(dominantColors)
    pyplot.figure(1)
    pyplot.subplot(1, 1, 1)
    pyplot.axis("off")
    pyplot.imshow(color_bar)
    pyplot.title("Color Bar")

    #Show all plots plotted
    pyplot.show()
    ''';
    
    
    # Dominant Color Information    
    '''
    print()
    print("Color Information :")
   
    for x in dominantColors:
        
        print(pprint.pformat(x))
        print()
    
    ''';

##################################################################################################################

# Dominant Color Determination
if galf == 0:
    
    rgb_lower = [0, 0, 0]
    rgb_mid1 = [150, 95, 60]
    rgb_mid2 = [224, 172, 105]
    rgb_higher = [255, 255, 255]

    skin_shades = {'dark' : [rgb_lower, rgb_mid1], 'mild' : [rgb_mid1, rgb_mid2], 'fair' : [rgb_mid2, rgb_higher]}

    decimal_lower = rgb_lower[0] * 256 * 256 + rgb_lower[1] * 256 + rgb_lower[2]
    decimal_higher = rgb_higher[0] * 256 * 256 + rgb_higher[1] * 256 + rgb_higher[2]
    
    convert_skintones = {}
    
    for shade in skin_shades:
    
        convert_skintones.update(
            {
                shade : 
            
                [
                    skin_shades[shade][0][0] * 256 * 256 + skin_shades[shade][0][1] * 256 + skin_shades[shade][0][2],
                    skin_shades[shade][1][0] * 256 * 256 + skin_shades[shade][1][1] * 256 + skin_shades[shade][1][2]
                ]
            }
        )

    unprocessed_dominant = extractDominantColor(skin, number_of_colors = 4, hasThresholding = True)
    dominantColors1 = []

    for clr in unprocessed_dominant:
    
        clr_decimal = 1 + int(clr['color'][0] * 256 * 256 + clr['color'][1] * 256 + clr['color'][2])
        
        if clr_decimal in range(decimal_lower, decimal_higher + 1):
        
            clr['decimal_color'] = clr_decimal
            dominantColors1.append(clr)

    skin_tones = []

    if len(dominantColors1) == 0:
    
        skin_tones.append('Unrecognized')
    
    else:
    
        for color in dominantColors1:
        
            for shade in convert_skintones:
            
                if color['decimal_color'] in range(convert_skintones[shade][0], convert_skintones[shade][1] + 1):
                
                    skin_tones.append(shade)

    #print(skin_tones)
    print(skin_tones[0])

