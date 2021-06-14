### Combining the top and side features:

import pandas as pd
import numpy as np
import os
import PIL
import seaborn as sns
import cv2




######################################################################################

def create_df():

  """This function creates and returns an emtpy Dataframe with the folloiwng columns"""


  df = pd.DataFrame(columns= ["Apple_ID", "Apple_type", "Area_top", "Circumference_top",
                                  "Solidity_top", "Shape_top", "Apple_height_top", "Apple_width_top", 
                                  "Extent_top","Circularity_top",
                                  "Convexity_top","Elongation_top",  
                                  "Area_side","Area_side_std", 
                                  "Circumference_side", "Circumference_side_std",  
                                  "Solidity_side", "Solidity_side_std" , 
                                  "Apple_height_side", "Apple_height_side_std",
                                  "Apple_width_side", "Apple_width_side_std", 
                                  "Extent_side", "Extent_side_std" , 
                                  "Circularity_side", "Circularity_side_std" ,  
                                  "Convexity_side", "Convexity_side_std", 
                                  "Elongation_side", "Elongation_side_std", 
                                  "Apple_volume"])

  return df 



######################################################################################

def Extract_feature(df_subset):

  """This function get a small subset of a dataframe, which belongs to a specific 
  apple_ID. The dataframe contains features of a specific apples taken by different 
  cameras (cam1 + (cam2, cam3 cam4 etc...)). Then it extracts the information from the top side
  (camera1) and vertical sides (other camers) of an apple, and build new features.
  Afterwards, it returns the data belong to a certain apple as a row. 
   
  """

  
  mydf = df_subset.reset_index(drop=True).copy()

  aaple_id = mydf.Apple_ID[0]
  apple_type = mydf.Apple_type[0]
  area_top = mydf.loc[mydf["Camera"] == "cam1", "Area"].tolist()
  circumference_top = mydf.loc[mydf["Camera"] == "cam1", "Circumference"].tolist()
  solidity_top = mydf.loc[mydf["Camera"] == "cam1", "Solidity"].tolist()
  shape_top = mydf.loc[mydf["Camera"] == "cam1", "Shape"].tolist()
  apple_height_top = mydf.loc[mydf["Camera"] == "cam1", "Apple_height"].tolist()
  apple_width_top = mydf.loc[mydf["Camera"] == "cam1", "Apple_width"].tolist()
  extent_top = mydf.loc[mydf["Camera"] == "cam1", "Extent"].tolist()
  circularity_top = mydf.loc[mydf["Camera"] == "cam1", "Circularity"].tolist()
  convexity_top = mydf.loc[mydf["Camera"] == "cam1", "Convexity"].tolist()
  elongation_top = mydf.loc[mydf["Camera"] == "cam1", "Elongation"].tolist()
  area_side = mydf[["Area"]].iloc[1:].mean(axis= 0).tolist() # it excludes the first row, which belongs to camera1
  area_side_std = mydf[["Area"]].iloc[1:].std(axis= 0).tolist()
  circumference_side = mydf[["Circumference"]].iloc[1:].mean(axis= 0).tolist()
  circumference_side_std = mydf[["Circumference"]].iloc[1:].std(axis= 0).tolist()
  solidity_side = mydf[["Solidity"]].iloc[1:].mean(axis= 0).tolist()
  solidity_side_std = mydf[["Solidity"]].iloc[1:].std(axis= 0).tolist()
  apple_height_side = mydf[["Apple_height"]].iloc[1:].mean(axis= 0).tolist()
  apple_height_side_std = mydf[["Apple_height"]].iloc[1:].std(axis= 0).tolist()
  apple_width_side = mydf[["Apple_width"]].iloc[1:].mean(axis= 0).tolist()
  apple_width_side_std = mydf[["Apple_width"]].iloc[1:].std(axis= 0).tolist()
  extent_side = mydf[["Extent"]].iloc[1:].mean(axis= 0).tolist()
  extent_side_std = mydf[["Extent"]].iloc[1:].std(axis= 0).tolist()
  circularity_side = mydf[["Circularity"]].iloc[1:].mean(axis= 0).tolist()
  circularity_side_std = mydf[["Circularity"]].iloc[1:].std(axis= 0).tolist()
  convexity_side = mydf[["Convexity"]].iloc[1:].mean(axis= 0).tolist()
  convexity_side_std = mydf[["Convexity"]].iloc[1:].std(axis= 0).tolist()
  elongation_side = mydf[["Elongation"]].iloc[1:].mean(axis= 0).tolist()
  elongation_side_std = mydf[["Elongation"]].iloc[1:].std(axis= 0).tolist()
  area = area_top[0]
  apple_volume = (area * apple_height_side[0])

  row = {"Apple_ID": mydf.Apple_ID[0], 
       "Apple_type" : mydf.Apple_type[0], 
       "Area_top": area_top[0],
       "Circumference_top": circumference_top[0],
       "Solidity_top": solidity_top[0],
       "Shape_top": shape_top[0],
       "Apple_height_top": apple_height_top[0],
       "Apple_width_top": apple_width_top[0],
       "Extent_top": extent_top[0],
       "Circularity_top": circularity_top[0],
       "Convexity_top": convexity_top[0],
       "Elongation_top": elongation_top[0],
       "Area_side": area_side[0],
       "Area_side_std": area_side_std[0],
       "Circumference_side": circumference_side[0],
       "Circumference_side_std": circumference_side_std[0],
       "Solidity_side": solidity_side[0],
       "Solidity_side_std": solidity_side_std[0],
       "Apple_height_side": apple_height_side[0],
       "Apple_height_side_std": apple_height_side_std[0],
       "Apple_width_side": apple_width_side[0],
       "Apple_width_side_std": apple_width_side_std[0],
       "Extent_side": extent_side[0],
       "Extent_side_std": extent_side_std[0],
       "Circularity_side": circularity_side[0],
       "Circularity_side_std": circularity_side_std[0],
       "Convexity_side": convexity_side[0],
       "Convexity_side_std": convexity_side_std[0],
       "Elongation_side": elongation_side[0],
       "Elongation_side_std": elongation_side_std[0],
       "Apple_volume": apple_volume,

  }

  return row 



######################################################################################

def combine_side_features(df):

  """gets a DataFrame, build the side and top features for each apple and the returns the final Dataframe"""

  df_final = create_df()


  for apple_type in df.Apple_type.unique():


    
    """Copy the subset of dataframe, which belongs to a cetrain apple_type, to a 
    new dataframe. Remember to add the copy() fucntion at the end, otherwise the new
    dataframe will not be a copy of the old one and it raises warning!"""

    
    df_new = df.loc[df["Apple_type"] == apple_type].reset_index(drop=True).copy()
    df_new = df_new.sort_values(by=['Camera'], ascending=True).reset_index(drop=True)


    if df_new.Camera[0] == "cam1":
    
      """Seperate the subset of dataframe, which belongs to camera 1."""
      df_cam_1 = df_new.loc[df_new["Camera"] == "cam1"].copy()
      for i in range (len(df_cam_1)):

        """In this part the program goes through the whole dataset belongs to the specfic apple_type
        and try to find the subset of datafame which belongs to a certian apple_ID. 
        """

        mydf = df_new.loc[df_new.Apple_ID == df_cam_1.Apple_ID[i]].copy()
        row = Extract_feature(mydf)
        df_final = df_final.append(row, ignore_index=True)
    else:
      print (f"{apple_type} does not have cam1 picture.")

  print ("Done! New DataFrame is ready, Enjoy further analysis!!!")

  return df_final




