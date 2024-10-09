#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats

from matplotlib import pyplot as plt
from typing import List, Dict, Any

import skimage.filters
import skimage.segmentation
import skimage.measure

def find_blebs(image: np.ndarray) -> List[np.ndarray]:
    """
    Find blebs in the given image.

    Args:
        image (np.ndarray): The input image.

    Returns:
        List[np.ndarray]: List of contours representing the blebs.
    """
    
    thresh = skimage.filters.threshold_triangle(image)
    binary_image = (image > thresh).astype(np.uint8)
    
    kernel = np.ones((5, 5), np.uint8) 
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)
    
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def analyze_bleb(contour: np.ndarray) -> dict:
    """
    Analyze the properties of a bleb contour.

    Args:
        contour (np.ndarray): The contour representing a bleb.

    Returns:
        dict: Dictionary containing the analyzed properties of the bleb.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    circularity = 0
    if perimeter:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
    ellipse = cv2.fitEllipse(contour)
    major_diameter, minor_diameter = ellipse[1]
    if minor_diameter > major_diameter:
        major_diameter, minor_diameter = minor_diameter, major_diameter

    return {
      "area": area,
      "circularity": circularity,
      "major_diameter": major_diameter,
      "minor_diameter": minor_diameter
    }


def display_blebs(original_frame: np.ndarray, bleb_array: np.ndarray) -> None:
    """
    Display the blebs on an image.

    Args:
        bleb_array (np.ndarray): The binary image containing the blebs.
    """
    contours, _ = cv2.findContours(bleb_array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = bleb_array.shape
    display_image = np.zeros((height, width, 3), np.uint8)
    
    bleb_number = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= 100:
            color = (0, 255, 0)
            cv2.drawContours(display_image, [cnt], 0, color, 2)


            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7 
            text_thickness = 2 
            cv2.putText(display_image, str(bleb_number), (x, y - 5), font, font_scale, (255, 255, 255), text_thickness)
            bleb_number += 1

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(original_frame)
    ax1.set_title('Original Frame')
    ax1.axis('off')
    ax2.imshow(bleb_array)
    ax2.set_title('Frame with Blebs')
    ax2.axis('off')
    ax3.imshow(display_image)
    ax3.set_title('Individual Blebs')
    ax3.axis('off')
    plt.show()


def write_to_excel(filename: str, data_dicts: List[Dict[str, List[Any]]], sheet_names: List[str]) -> None:
    """
    Write data to an Excel file.

    Args:
        filename (str): The name of the Excel file.
        data_dicts (List[Dict[str, List[Any]]]): List of dictionaries containing the data to be written.
        sheet_names (List[str]): List of sheet names corresponding to each data dictionary.
    """
    with pd.ExcelWriter(filename) as writer:
        for i, df in enumerate(data_dicts):
            max_length = max(len(df[a]) for a in df)

            for col in df:
                df[col] += [np.nan] * (max_length - len(df[col]))

            df_temp = pd.DataFrame(df)
            df_temp.to_excel(writer, sheet_name=sheet_names[i], index=False)


def make_box_plot_with_significance(df: List[Dict[str, List[Any]]], df_name: str, save_fig=False) -> None:
    """
    Generate a box plot with significance levels for the given data.

    Args:
        df (List[Dict[str, List[Any]]]): A list of dictionaries containing the data.
        df_name (str): The name of the data.

    Returns:
        None
    """
    max_length = max(len(df[a]) for a in df)

    for col in df:
        df[col] += [np.nan] * (max_length - len(df[col]))

    df_temp = pd.DataFrame(df)
    
    sns.boxplot(
        x = "variable",
        y = "value",
        showmeans=True,   
        data=df_temp.melt(var_name="variable", value_name="value"),
    )

    plt.xlabel("Concentration")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title(f"Distribution of {df_name}")

    # Perform t-tests
    for j in range(1, len(df)):
        _, p_value = stats.ttest_ind(df_temp["control"], df_temp[df_temp.columns[j]], nan_policy='omit')
        if p_value < 0.05:
            if p_value < 0.01:
                if p_value < 0.001:
                    significance = "***"
                else:
                    significance = "**"
            else:
                significance = "*"
        else:
            significance = "n.s."
        
        # Add significance level to the plot            
        y_max = max(max(df_temp["control"]), max(df_temp[df_temp.columns[j]]))
        plt.text(j, y_max, significance, ha='center')
    
    sns.despine(top=True, right=True)
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{df_name}.png")
    plt.show()