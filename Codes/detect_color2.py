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

### Seperating the LAB components 
def add_new_columns(df):

  """Add new columns to data frame for LAB components"""
  df["Color1_L"] = None
  df["Color1_A"] = None
  df["Color1_B"] = None
  df["Color2_L"] = None
  df["Color2_A"] = None
  df["Color2_B"] = None
  df["Color3_L"] = None
  df["Color3_A"] = None
  df["Color3_B"] = None
  df["Color4_L"] = None
  df["Color4_A"] = None
  df["Color4_B"] = None
  df["Color5_L"] = None
  df["Color5_A"] = None
  df["Color5_B"] = None

  return df 



#####################################################################################################

def seperate_LAB_components (df):

  df = add_new_columns(df)

  """This function creats a dictionary for LAB components"""
  
  color_columns = {'Color1': ['Color1_L', 'Color1_A', 'Color1_B'], 
                  'Color2': ['Color2_L', 'Color2_A', 'Color2_B'], 
                  'Color3': ['Color3_L', 'Color3_A', 'Color3_B'], 
                  'Color4': ['Color4_L', 'Color4_A', 'Color4_B'], 
                  'Color5': ['Color5_L', 'Color5_A', 'Color5_B']}

  

  """In the following program col is the name of the color and lab_color is the name of 
  the new LAB components"""

  for col, lab_color in color_columns.items():
    print (col, "-->", lab_color)
    for i in range(len(df)):
      L = lab_color[0]
      A = lab_color[1]
      B = lab_color[2]
      df[L][i] = df[col][i][0][0][0]
      df[A][i] = df[col][i][0][0][1]
      df[B][i] = df[col][i][0][0][2]
  
  df = df.drop(['Color1', 'Color2', 'Color3', 'Color4','Color5'], axis = 1 )
  return df

  print ("Done!!!")



#####################################################################################################
  
def create_df_color_final():

  """This function creates a new Dataframe """

  df = pd.DataFrame(columns= ["Apple_ID", "Apple_type", 
                                    "Color1_L_top", "Color1_A_top", "Color1_B_top", 
                                    "Color2_L_top", "Color2_A_top", "Color2_B_top", 
                                    "Color3_L_top", "Color3_A_top", "Color3_B_top",
                                    "Color4_L_top", "Color4_A_top", "Color4_B_top",
                                    "Color5_L_top", "Color5_A_top", "Color5_B_top",
                                    "Color1_percent_top", "Color2_percent_top", 
                                    "Color3_percent_top",
                                    "Color4_percent_top", "Color5_percent_top",
                                    "Color1_L_side", "Color1_A_side", "Color1_B_side",  
                                    "Color2_L_side", "Color2_A_side", "Color2_B_side", 
                                    "Color3_L_side", "Color3_A_side", "Color3_B_side",
                                    "Color4_L_side", "Color4_A_side", "Color4_B_side", 
                                    "Color5_L_side", "Color5_A_side", "Color5_B_side", 
                                    "Color1_percent_side", "Color2_percent_side", 
                                    "Color3_percent_side",
                                    "Color4_percent_side", "Color5_percent_side",
  ])

  return df 



#####################################################################################################

def Extract_top_and_side_color(df):

  mydf = df.reset_index(drop=True).copy()
               

  apple_id = mydf.Apple_ID[0]
  apple_type = mydf.Apple_type[0]
  color1_L_top = mydf.loc[mydf["Camera"] == "cam1", "Color1_L"].tolist()
  color1_A_top = mydf.loc[mydf["Camera"] == "cam1", "Color1_A"].tolist()
  color1_B_top = mydf.loc[mydf["Camera"] == "cam1", "Color1_B"].tolist()
  color2_L_top = mydf.loc[mydf["Camera"] == "cam1", "Color2_L"].tolist()
  color2_A_top = mydf.loc[mydf["Camera"] == "cam1", "Color2_A"].tolist() 
  color2_B_top = mydf.loc[mydf["Camera"] == "cam1", "Color2_B"].tolist()
  color3_L_top = mydf.loc[mydf["Camera"] == "cam1", "Color3_L"].tolist()
  color3_A_top = mydf.loc[mydf["Camera"] == "cam1", "Color3_A"].tolist()
  color3_B_top = mydf.loc[mydf["Camera"] == "cam1", "Color3_B"].tolist()
  color4_L_top = mydf.loc[mydf["Camera"] == "cam1", "Color4_L"].tolist()
  color4_A_top = mydf.loc[mydf["Camera"] == "cam1", "Color4_A"].tolist()
  color4_B_top = mydf.loc[mydf["Camera"] == "cam1", "Color4_B"].tolist()
  color5_L_top = mydf.loc[mydf["Camera"] == "cam1", "Color5_L"].tolist()
  color5_A_top = mydf.loc[mydf["Camera"] == "cam1", "Color5_A"].tolist()
  color5_B_top = mydf.loc[mydf["Camera"] == "cam1", "Color5_B"].tolist()
  color1_percent_top = mydf.loc[mydf["Camera"] == "cam1", "Color1_percent"].tolist()
  color2_percent_top = mydf.loc[mydf["Camera"] == "cam1", "Color2_percent"].tolist()
  color3_percent_top = mydf.loc[mydf["Camera"] == "cam1", "Color3_percent"].tolist()
  color4_percent_top = mydf.loc[mydf["Camera"] == "cam1", "Color4_percent"].tolist()
  color5_percent_top = mydf.loc[mydf["Camera"] == "cam1", "Color5_percent"].tolist()
  color1_L_side = mydf.loc[mydf["Camera"] == "cam_side", "Color1_L"].tolist()
  color1_A_side = mydf.loc[mydf["Camera"] == "cam_side", "Color1_A"].tolist()
  color1_B_side = mydf.loc[mydf["Camera"] == "cam_side", "Color1_B"].tolist()
  color2_L_side = mydf.loc[mydf["Camera"] == "cam_side", "Color2_L"].tolist()
  color2_A_side = mydf.loc[mydf["Camera"] == "cam_side", "Color2_A"].tolist()
  color2_B_side = mydf.loc[mydf["Camera"] == "cam_side", "Color2_B"].tolist()
  color3_L_side = mydf.loc[mydf["Camera"] == "cam_side", "Color3_L"].tolist()
  color3_A_side = mydf.loc[mydf["Camera"] == "cam_side", "Color3_A"].tolist()
  color3_B_side = mydf.loc[mydf["Camera"] == "cam_side", "Color3_B"].tolist()
  color4_L_side = mydf.loc[mydf["Camera"] == "cam_side", "Color4_L"].tolist()
  color4_A_side = mydf.loc[mydf["Camera"] == "cam_side", "Color4_A"].tolist()
  color4_B_side = mydf.loc[mydf["Camera"] == "cam_side", "Color4_B"].tolist()
  color5_L_side = mydf.loc[mydf["Camera"] == "cam_side", "Color5_L"].tolist()
  color5_A_side = mydf.loc[mydf["Camera"] == "cam_side", "Color5_A"].tolist()
  color5_B_side = mydf.loc[mydf["Camera"] == "cam_side", "Color5_B"].tolist()
  color1_percent_side = mydf.loc[mydf["Camera"] == "cam_side", "Color1_percent"].tolist()
  color2_percent_side = mydf.loc[mydf["Camera"] == "cam_side", "Color2_percent"].tolist()
  color3_percent_side = mydf.loc[mydf["Camera"] == "cam_side", "Color3_percent"].tolist()
  color4_percent_side = mydf.loc[mydf["Camera"] == "cam_side", "Color4_percent"].tolist()
  color5_percent_side = mydf.loc[mydf["Camera"] == "cam_side", "Color5_percent"].tolist()

  row = {"Apple_ID": apple_id, 
             "Apple_type" : apple_type,     
             "Color1_L_top" : color1_L_top[0], 
             "Color1_A_top" : color1_A_top[0],
             "Color1_B_top" : color1_B_top[0],
             "Color2_L_top" : color2_L_top[0], 
             "Color2_A_top" : color2_A_top[0],
             "Color2_B_top" : color2_B_top[0],
             "Color3_L_top" : color3_L_top[0], 
             "Color3_A_top" : color3_A_top[0],
             "Color3_B_top" : color3_B_top[0],
             "Color4_L_top" : color4_L_top[0], 
             "Color4_A_top" : color4_A_top[0],
             "Color4_B_top" : color4_B_top[0],
             "Color5_L_top" : color5_L_top[0], 
             "Color5_A_top" : color5_A_top[0],
             "Color5_B_top" : color5_B_top[0],
             "Color1_percent_top" : color1_percent_top[0],
             "Color2_percent_top" : color2_percent_top[0],
             "Color3_percent_top" : color3_percent_top[0],
             "Color4_percent_top" : color4_percent_top[0],
             "Color5_percent_top" : color5_percent_top[0],
             "Color1_L_side" : color1_L_side[0], 
             "Color1_A_side" : color1_A_side[0],
             "Color1_B_side" : color1_B_side[0],
             "Color2_L_side" : color2_L_side[0],
             "Color2_A_side" : color2_A_side[0],
             "Color2_B_side" : color2_B_side[0],
             "Color3_L_side" : color3_L_side[0],
             "Color3_A_side" : color3_A_side[0],
             "Color3_B_side" : color3_B_side[0],
             "Color4_L_side" : color4_L_side[0],
             "Color4_A_side" : color4_A_side[0], 
             "Color4_B_side" : color4_B_side[0],
             "Color5_L_side" : color5_L_side[0],
             "Color5_A_side" : color5_A_side[0],
             "Color5_B_side" : color5_B_side[0],
             "Color1_percent_side" : color1_percent_side[0],
             "Color2_percent_side" : color2_percent_side[0],
             "Color3_percent_side" : color3_percent_side[0],
             "Color4_percent_side" : color4_percent_side[0],
             "Color5_percent_side" : color5_percent_side[0],
                                  }
  return row



#####################################################################################################

def get_final_color (df):

  df_color_final = create_df_color_final() 

  for apple_type in df.Apple_type.unique():
    
    df_new = df.loc[df["Apple_type"] == apple_type].reset_index(drop=True).copy()
    df_new = df_new.sort_values(by=['Camera'], ascending=True).reset_index(drop=True)

    if df_new.Camera[0] == "cam1":

      df_cam_1 = df_new.loc[df_new["Camera"] == "cam1"].copy()

      for i in range (len(df_cam_1)):

        mydf = df_new.loc[df_new.Apple_ID == df_cam_1.Apple_ID[i]].copy()
        row = Extract_top_and_side_color(mydf)
        df_color_final = df_color_final.append(row, ignore_index=True)

    else:

      print (f"{apple_type} does not have top camera! ")

  print ("Well done! you made it, Good luck with your analysis!")

  return df_color_final


