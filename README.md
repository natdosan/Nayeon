# ONGOING PROJECT


# Nayeon - A project to analyze statistical trends in numerical data


This is a simple script that generates a regression model for a dataset. For this example, I used customer information for a restaurant. 

In addition to visualization, this script will also predict a chosen variable based on other variables. Intutitively, you will want to predict a dependent variable based on dependent variables.
  
   For example, if we wanted to see if there was a correlation between age and order total, we would set 
   ```
   predict = "Order total"
   ```
   and set our independent variable:
   ```
   independent = "Age"
   ```
   Note: these assignments must match exactly with the column titles in the csv/txt file

# How it works
I used sklearn for training and testing models. For visualization, pyplot from the matplotlib library served as a convenient way of graphing the real data. 
