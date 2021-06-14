# Shape detector function
import pandas as pd
import numpy as np
import os
import PIL
import seaborn as sns
import cv2

"""In this document the Shape features for all the apple images are extracted """


######################################################################################

def detectShape(cnt):

    """This function gets a contour and returns the corresponding shape for that."""
    
    c = cnt 
    shape = 'unknown'
    # calculate perimeter using
    peri = cv2.arcLength(c, True)
    # apply contour approximation and store the result in vertices
    vertices = cv2.approxPolyDP(c, 0.04 * peri, True)

    # If the shape it triangle, it will have 3 vertices
    if len(vertices) == 3:
        shape = 'triangle'

    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(vertices) == 4:
        # using the boundingRect method calculate the width and height
        # of enclosing rectange and then calculte aspect ratio

        x, y, width, height = cv2.boundingRect(vertices)
        aspectRatio = float(width) / height

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            shape = "square"
        else:
            shape = "rectangle"

    # if the shape is a pentagon, it will have 5 vertices
    elif len(vertices) == 5:
        shape = "pentagon"

    # otherwise, we assume the shape is a circle
    else:
        shape = "circle"

    # return the name of the shape
    return shape



######################################################################################

def Get_area_shape_index(contours):

  """This line goes through all the detected contours, calculates their shapes and their areas,
    and returns the second largest contour which belongs to an apple."""

  df = pd.DataFrame(index=range(0, len(contours)))

  df["contour_area"] = None
  df["contour_index"] = None
  df["shape"] = None
  

  # cX and cY are the x and y coordinates of apple centeral point
  cX= 0 
  cY= 0
  shape = 0

  for index, c in enumerate(contours):
    
    
    df["contour_area"][index] = cv2.contourArea(c)
    df["contour_index"][index] = index
    df["shape"][index] = detectShape(c)
      

  """among all the detected contours the first and the largest one is always the frame of the 
  picture which usually has a reactangular shape. We do not need that. Therefore using the following 
  lines first the dataframe is sorted according to the contour_area, then the second largets value
  which, belongs to apple contour would be extracted from the Dataframe.  """
  
  df.sort_values(by=['contour_area'], ascending=False, inplace=True)
  contour_area , contour_index , shape = df.iloc[1] # chose the second largest value of contours which belongs to apple

  return contour_area , contour_index , shape 

  """compute the moment of contour
  From moment we can calculte area, centroid etc
  The center or centroid can be calculated as follows"""

  M = cv2.moments(contours[max_index_incontours])
  cX = int(M['m10'] / M['m00'])
  cY = int(M['m01'] / M['m00'])



######################################################################################
  
def add_columns(df):

  """This function creates and return an empty Dataframe."""  
    
  df["Area"] = None 
  df["Circumference"] = None 
  df["Solidity"] = None  
  df["Shape"] = None 
  df["Apple_height"] = None
  df["Apple_width"] = None
  df["Extent"] = None
  df["Circularity"] = None
  df["Convexity"] = None
  df["Elongation"] = None 
  df["Raw_image"] = None

  return df



######################################################################################

def First_image_processing (pic):

  """This function gets an image and returns the contours of the detected objects."""
  
  # Conver the RGB sacle to gray scale
  gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

  # Blur the picture to reduce the noises
  blured = cv2.GaussianBlur(gray,(51,51),cv2.BORDER_DEFAULT)

  # Thresholding means revealing the shapes in the image
  retval,threshold3 = cv2.threshold(blured,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

  #find contours or the closed curves in the picture. A contour is a closed curve 
  contours, hierarchy = cv2.findContours(threshold3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  return contours



######################################################################################

def shape_features (df):

  """This function gets a Dataframe and extract the shape features for each image. """
  

  df = add_columns(df)   # create an empty Dataframe

  for i in range (0, len(df.Image)):

    pic = df.Image[i]
    raw_image = pic.copy()  
    contours = First_image_processing (pic)

    #print (contours)
    Apple_area , Apple_index , Apple_shape  = Get_area_shape_index(contours)

    
    """The following line calculate the width and height of the contour.
    x and y are the coordinate of the first corner of the box around the contour. 
    w and h are width and height of the detected object."""

  #Properties of the Apple Contours       
    x, y, w, h = cv2.boundingRect(contours[Apple_index])
    modified_image = raw_image[y:y+h,x:x+w] # Croped raw image for color detection

    """ Orientation is the angle at which object is directed. 
    Following method also gives the center of the ellipse, Major Axis and Minor Axis lengths, 
    and the angle of orientation """

    (x,y),(minor_axis,major_axis),angle = cv2.fitEllipse(contours[Apple_index])
    
    hull = cv2.convexHull(contours[Apple_index])
    hullArea = cv2.contourArea(hull)
    hull_Circumference = cv2.arcLength(hull,True)
    Apple_Circumference = cv2.arcLength(contours[Apple_index],True)
    
    df["Area"][i] = Apple_area
    df["Circumference"][i] = Apple_Circumference
    df["Solidity"][i] =  Apple_area / float(hullArea)
    df["Shape"][i] = Apple_shape
    df["Apple_height"][i] = h
    df["Apple_width"][i] = w
    df["Extent"][i] = Apple_area / (float(w*h))
    df["Circularity"][i] = (4 * np.pi * Apple_area)/ np.power(Apple_Circumference, 2)
    df["Convexity"][i] = hull_Circumference / Apple_Circumference
    df["Elongation"][i] = np.sqrt(1-(minor_axis/major_axis)**2)
    df["Raw_image"][i] = modified_image

  df = df.drop(["Image"], axis = 1)
  print ("Well done! the shape features for every image is extracted.")

  return df  





  

