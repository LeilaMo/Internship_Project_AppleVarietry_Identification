#!/usr/bin/env python
###color detection functions and it works with DataFrame before adding top and side features
import pandas as pd
import numpy as np
import os
import PIL
import seaborn as sns
import cv2
import math
from sklearn.cluster import KMeans
from collections import Counter
import skimage
from skimage.color import rgb2lab, deltaE_cie76


"""
1 - Seperate top and side pictures
2- put side pictures for an individual apple together
3- Join 1 and 2
4- Extrcat 5 most dominant colors 

"""


#####################################################################################################

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):

  """This function put the pictures with different size, side by side and 
  return it as a new picture. """

  h_min = min(im.shape[0] for im in im_list)
  im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min),
                               interpolation=interpolation)
                      for im in im_list]
  return cv2.hconcat(im_list_resize)



#####################################################################################################

def Add_side_pictures_for_one_apple(df):

  """This function gets a subset related to a specific Apple_ID, put the 
  side pictures together and returns a row"""

  mydf = df.reset_index(drop=True).copy()

  camera = "cam_side"      
  apple_id = mydf.Apple_ID[0]
  apple_type = mydf.Apple_type[0]

  pictures_list = mydf["Raw_image"].tolist()
  apple_side_picture = hconcat_resize_min(pictures_list)

  row = {"Camera" : camera , 
         "Apple_type" : apple_type, 
         "Raw_image" : apple_side_picture, 
         "Apple_ID" : apple_id}


  return row 



#####################################################################################################

def add_side_pictures_together (df):

  """This function gets a Dataframe, seperate the pictures taken by cam1 (df_top) and other 
  cameras (df_side). For each apple put the side pictures together
  and add it to a new Dataframe called df_side_final. When the process is done 
  for all the apples, the function concatenates both df_top and df_side_final 
  and returns it"""
  
  
  
  df_top = df.loc[df["Camera"] == "cam1"].reset_index(drop=True).copy() # extract the pictures taken by cam1
  df_side = df.loc[df["Camera"] != "cam1"].reset_index(drop=True).copy() # extract the pictures taken by other cameras

  df_side_final = pd.DataFrame() # create a new Dataframe



  # This part just analyse the dataframe related to side pictures

  for apple_type in df_side.Apple_type.unique():

    """ """
    print (f"{apple_type} is in process...")

    df_new = df_side.loc[df_side["Apple_type"] == apple_type].reset_index(drop=True).copy() #extract all the pictures of apple_type_x
    df_new = df_new.sort_values(by=['Camera'], ascending=True).reset_index(drop=True) # sort them by camera and reset the index 

    apple_id = df_new.Apple_ID.unique() # extract the apple_ID 

    for id in apple_id:

      """In the next step the pictures related to a specific Apple_ID are 
      extracted and sent to Add_side_pictures_for_one_apple """


      mydf = df_new.loc[df_new.Apple_ID == id ].copy()
      row = Add_side_pictures_for_one_apple(mydf)
      df_side_final = df_side_final.append(row, ignore_index=True)
    
  print ("Huraaaaayyyy!!! you made it, Good luck with your analysis!")

  df_top_side = pd.concat([df_top, df_side_final], ignore_index=True)

  return df_top_side



#####################################################################################################

def RGB2HEX(color):

  """This function convert the RGB color to a Hexadecimal color code""" 
  return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def RGB2LAB(color):

  """This function convert the RGB color to L * A * B color and the result 
  is a 3D array, since skimage.color.rgb2lab only accepts 3D arraies"""
  
  return skimage.color.rgb2lab(np.array(np.ones((1, 1, 3)) * color/255))



#####################################################################################################

def get_color_df (labels, cluster_centers, number_of_colors):

  """This function recives labels, cluster_centers and number_of_colors"""


  df = pd.DataFrame(index=range(0, number_of_colors))

  df["color_index"] = None 
  df["rgb_color"] = None 
  df["counts"] = None
  df["percentage"] = None 
  df["hex_color_code"] = None 
  df["lab_color"] = None 


  label_index = np.arange(0,len(np.unique(labels)) + 1) 
  colors = cluster_centers  # center_colors contains the most dominant colors in RGB

  hist, _= np.histogram(labels, bins= label_index) # calculates the 
  counts = hist.copy() # the number of 
  hist = hist.astype("float")
  hist = [i/hist.sum() for i in hist]

  hex_colors = [RGB2HEX(i) for i in colors]


  df["color_index"] = [i  for i in range(len(df))]   
  df["rgb_color"] = [i  for i in colors]  
  df["counts"] = counts
  df["percentage"] = hist # contains color percentage
  df["hex_color_code"] = hex_colors
  df["lab_color"] = [RGB2LAB(i)  for i in colors]
  df.sort_values(by=['percentage'], ascending=False, inplace=True)
  df.reset_index(drop=True, inplace=True)

  return df

  for i in range(len(df)):
        print(df.rgb_color[i], "{:0.2f}%".format(df.percentage[i] * 100))



#####################################################################################################
        
def DropBackgroundColor (df):
  """This function recieve a dataframe, and removes the lightgray or white color
  from the background of the image. """


  for index, color in enumerate(df.rgb_color):

    if color[0] >= 235 and color[1] >= 235 and color[2] >= 235:
      df.drop([index], inplace = True)
      df.reset_index(drop=True, inplace=True)

  return df



#####################################################################################################

def add_columns_to_dataFrame(df):
    
  # This function creates empty DataFrame coloumns
  df["Color1"] = None 
  df["Color2"] = None 
  df["Color3"] = None 
  df["Color4"] = None 
  df["Color5"] = None 
  df["Color1_percent"] = None
  df["Color2_percent"] = None
  df["Color3_percent"] = None
  df["Color4_percent"] = None
  df["Color5_percent"] = None


  return df



#####################################################################################################

def color_detection(df):

  """This function recives a Dataframe, and extract 5 most dominat colors of the pictures """

  df = add_columns_to_dataFrame(df) 

  print ("5 most dominant colors are going to be extrcated! if you want more or less please change the code.")
  print ("Calculations are running, be patient...")
  for i in range(0, len(df)):

    print (i)

    modified_image = df.Raw_image[i]
    # reshape the image to be a list of pixels, an array of 3 dimensioal pixels
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)


    number_of_colors = 6   # the number of dominant colors to be detected in the image
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    df2 = get_color_df(labels, clf.cluster_centers_, number_of_colors)

      # The program removes the background color and will return 6 most dominant colors in the picture
    df2 = DropBackgroundColor(df2) 
    df2["percentage"] = df2["percentage"] * 100 


    """The extracted DataFrame needs to be processed again, in order to exract the most relavant 
      information such as percentage, color code etc..."""

    
    df["Color1"][i] = df2.lab_color[0] 
    df["Color2"][i] = df2.lab_color[1] 
    df["Color3"][i] = df2.lab_color[2] 
    df["Color4"][i] = df2.lab_color[3]
    df["Color5"][i] = df2.lab_color[4]
    df["Color1_percent"][i] = df2.percentage[0]
    df["Color2_percent"][i] = df2.percentage[1]
    df["Color3_percent"][i] = df2.percentage[2]
    df["Color4_percent"][i] = df2.percentage[3]
    df["Color5_percent"][i] = df2.percentage[4]

    
  df = df.drop("Raw_image", axis = 1)
  
  print ("Done!")
  
  return df 

